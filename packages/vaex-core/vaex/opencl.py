import numpy as np
import threading
import logging
import warnings

import vaex.serialize
from .expression import FunctionSerializableJit
from . import expresso



logger = logging.getLogger("vaex.webserver.tornado")


class ExpressionStringMetal(expresso.ExpressionString):
    def pow(self, left, right):
        return "pow({left}, {right})".format(left=left, right=right)


def node_to_cpp(node, pretty=False):
    return ExpressionStringMetal(pretty=pretty).visit(node)


import faulthandler
faulthandler.enable()

@vaex.serialize.register
class FunctionSerializableOpenCL(FunctionSerializableJit):
    device = None
    def compile(self):
        try:
            import pyopencl as cl
        except ImportError:
            logging.error("Failure to import pyopencl, please install pyopencl")
            raise
        import objc
        dtype_out = vaex.dtype(self.return_dtype).numpy
        if dtype_out.name == "float64":
            dtype_out = np.dtype("float32")
            warnings.warn("Casting output from float64 to float32 since Metal does not support float64")
        ast_node = expresso.parse_expression(self.expression)
        cppcode = node_to_cpp(ast_node)
        typemap = {'float32': 'float',
                   'float64': 'float'}  # we downcast!
        for name in vaex.array_types._type_names_int:
            typemap[name] = f'{name}_t'
        typenames = [typemap[dtype.name] for dtype in self.argument_dtypes]
        metal_args = [f'__global const {typename} *{name}_array' for i, (typename, name) in
                      enumerate(zip(typenames, self.arguments))]
        code_get_scalar = [f'    {typename} {name} = {name}_array[gid];\n' for typename, name, in zip(typenames, self.arguments)]
        sourcecode = '''
float arctan2(float y, float x) {
    return atan2(y, x);
}

__kernel void vaex_kernel(%s,
                        __global %s *vaex_output) {
    int gid = get_global_id(0);
%s
    vaex_output[gid] = %s;
}
''' % (', '.join(metal_args), typemap[dtype_out.name], ''.join(code_get_scalar), cppcode) # typemap[self.dtype_out],
        if self.verbose:
            print('Generated code:\n' + sourcecode)
        with open('opencl-vaex.cpp', 'w') as f:
            logger.info(f'Wrote metal shader to {f.name}')
            f.write(sourcecode)


        storage = threading.local()

        self.context = cl.create_some_context(interactive=False)
        mf = cl.mem_flags

        self.program = cl.Program(self.context, sourcecode).build()

        def wrapper(*args):
            queue = cl.CommandQueue(self.context)
            args = [vaex.array_types.to_numpy(ar) for ar in args]
            nannies = []
            def getbuf(name, value=None, dtype=np.dtype("float32"), N=None, write=False):
                buf = getattr(storage, name, None)
                if value is not None:
                    N = len(value)
                    dtype = value.dtype
                if dtype.name == "float64":
                    warnings.warn("Casting input argument from float64 to float32 since Metal does not support float64")
                    dtype = np.dtype("float32")
                nbytes = N * dtype.itemsize
                if buf is not None and buf.size != nbytes:
                    # doesn't match size, create a new one
                    buf = None
                # create a buffer
                if buf is None:
                    print("create", nbytes)
                    buf = cl.Buffer(self.context, mf.READ_ONLY if not write else mf.READ_WRITE, size=nbytes)
                    setattr(storage, name, buf)
                # copy data to buffer
                if value is not None:
                    data = value.astype(dtype, copy=False)
                    mv = memoryview(data)
                    ret = cl.enqueue_copy(queue, buf, mv)
                    nannies.append(ret)
                    # print(ret)
                return buf
            input_buffers = [getbuf(name, chunk) for name, chunk in zip(self.arguments, args)]
            output_buffer = getbuf('vaex_output', N=len(args[0]), dtype=dtype_out, write=True)
            buffers = input_buffers + [output_buffer]

            self.program.vaex_kernel(queue, args[0].shape, None, *buffers)
            
            result = np.empty(shape=(len(args[0])), dtype=dtype_out)
            cl.enqueue_copy(queue, result, output_buffer)
            queue.finish()
            return result
        return wrapper

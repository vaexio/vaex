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
class FunctionSerializableMetal(FunctionSerializableJit):
    device = None
    def compile(self):
        try:
            import Metal
        except ImportError:
            logging.error("Failure to import Metal, please install pyobjc-framework-Metal")
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
        metal_args = [f'const device {typename} *{name}_array [[buffer({i})]]' for i, (typename, name) in
                      enumerate(zip(typenames, self.arguments))]
        code_get_scalar = [f'    {typename} {name} = {name}_array[id];\n' for typename, name, in zip(typenames, self.arguments)]
        sourcecode = '''
#include <metal_stdlib>
using namespace metal;

float arctan2(float y, float x) {
    return atan2(y, x);
}

template<typename T>
T where(bool condition, T y, T x) {
    return condition ? x : y;
}
kernel void vaex_kernel(%s,
                        device %s *vaex_output [[buffer(%i)]],
                        uint id [[thread_position_in_grid]]) {
%s
    vaex_output[id] = %s;
}
''' % (', '.join(metal_args), typemap[dtype_out.name], len(metal_args), ''.join(code_get_scalar), cppcode) # typemap[self.dtype_out],
        if self.verbose:
            print('Generated code:\n' + sourcecode)
        with open('test.metal', 'w') as f:
            print(f'Write to {f.name}')
            f.write(sourcecode)


        storage = threading.local()
        lock = threading.Lock()

        # following https://developer.apple.com/documentation/metal/basic_tasks_and_concepts/performing_calculations_on_a_gpu?language=objc
        self.device = Metal.MTLCreateSystemDefaultDevice()
        opts = Metal.MTLCompileOptions.new()
        self.library = self.device.newLibraryWithSource_options_error_(sourcecode, opts, objc.NULL)
        if self.library[0] is None:
            msg = f"Error compiling: {sourcecode}, sourcecode"
            logger.error(msg)
            raise RuntimeError(msg)
        kernel_name = "vaex_kernel"
        self.vaex_kernel = self.library[0].newFunctionWithName_(kernel_name)
        desc = Metal.MTLComputePipelineDescriptor.new()
        desc.setComputeFunction_(self.vaex_kernel)
        state = self.device.newComputePipelineStateWithDescriptor_error_(desc, objc.NULL)
        command_queue = self.device.newCommandQueue()

        def wrapper(*args):
            args = [vaex.array_types.to_numpy(ar) for ar in args]
            def getbuf(name, value=None, dtype=np.dtype("float32"), N=None):
                buf = getattr(storage, name, None)
                if value is not None:
                    N = len(value)
                    dtype = value.dtype
                if dtype.name == "float64":
                    warnings.warn("Casting input argument from float64 to float32 since Metal does not support float64")
                    dtype = np.dtype("float32")
                nbytes = N * dtype.itemsize
                if buf is not None and buf.length() != nbytes:
                    # doesn't match size, create a new one
                    buf = None
                # create a buffer
                if buf is None:
                    buf = self.device.newBufferWithLength_options_(nbytes, 0)
                    setattr(storage, name, buf)
                # copy data to buffer
                if value is not None:
                    mv = buf.contents().as_buffer(buf.length())
                    buf_as_numpy = np.frombuffer(mv, dtype=dtype)
                    buf_as_numpy[:] = value.astype(dtype, copy=False)
                return buf
            input_buffers = [getbuf(name, chunk) for name, chunk in zip(self.arguments, args)]
            output_buffer = getbuf('vaex_output', N=len(args[0]), dtype=dtype_out)
            buffers = input_buffers + [output_buffer]
            command_buffer = command_queue.commandBuffer()
            encoder = command_buffer.computeCommandEncoder()
            encoder.setComputePipelineState_(state)
            for i, buf in enumerate(buffers):
                encoder.setBuffer_offset_atIndex_(buf, 0, i)
            nitems = len(args[0])
            tpgrid = Metal.MTLSize(width=nitems, height=1, depth=1)
            # state.threadExecutionWidth() == 32 on M1 max
            # state.maxTotalThreadsPerThreadgroup() == 1024 on M1 max
            tptgroup = Metal.MTLSize(width=state.threadExecutionWidth(), height=state.maxTotalThreadsPerThreadgroup()//state.threadExecutionWidth(), depth=1)
            # this is simpler, and gives the same performance
            # tptgroup = Metal.MTLSize(width=1, height=1, depth=1)
            encoder.dispatchThreads_threadsPerThreadgroup_(tpgrid, tptgroup)
            encoder.endEncoding()
            command_buffer.commit()
            command_buffer.waitUntilCompleted()

            output_buffer_py = output_buffer.contents().as_buffer(output_buffer.length())
             # do we needs .copy() ?
            result = np.frombuffer(output_buffer_py, dtype=dtype_out)
            return result
        return wrapper

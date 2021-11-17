import numpy as np
import threading

import vaex.serialize
from .expression import FunctionSerializableJit
from . import expresso


class ExpressionStringMetal(expresso.ExpressionString):
    def pow(self, left, right):
        return "pow({left}, {right})".format(left=left, right=right)


def node_to_cpp(node, pretty=False):
    return ExpressionStringMetal(pretty=pretty).visit(node)


@vaex.serialize.register
class FunctionSerializableMetal(FunctionSerializableJit):
    def compile(self):
        import runmetal
        ast_node = expresso.parse_expression(self.expression)
        cppcode = node_to_cpp(ast_node)
        typemap = {'float32': 'float',
                   'float64': 'float'}  # we downcast!
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
kernel void vaex_kernel(%s,
                        device float *vaex_output [[buffer(%i)]],
                        uint id [[thread_position_in_grid]]) {
%s
    vaex_output[id] = %s;
}
''' % (', '.join(metal_args), len(metal_args), ''.join(code_get_scalar), cppcode)
        if self.verbose:
            print('Generated code:\n' + sourcecode)
        with open('test.metal', 'w') as f:
            print(f'Write to {f.name}')
            f.write(sourcecode)

        pm = runmetal.PyMetal()
        pm.opendevice()
        pm.openlibrary(sourcecode)
        vaex_kernel = pm.getfn("vaex_kernel")

        storage = threading.local()
        lock = threading.Lock()

        def wrapper(*args):
            chunks = args = [arg * 1.0 for arg in args]

            def getbuf(name, value=None, dtype=np.float32(), N=None):
                buf = getattr(storage, name, None)
                if value is not None:
                    N = len(value)
                    dtype = value.dtype
                if buf is not None and buf.length() != dtype.itemsize * N:
                    buf = None
                # buf = None  # if we don't reused buffers, we can move the lock
                if buf is None:
                    if value is not None:
                        value = value.astype(np.float32)
                    else:
                        value = np.zeros(N, dtype=dtype)
                        print("length", N)
                    buf = pm.numpybuffer(value)
                    setattr(storage, name + "_array", value)
                    # buf = pm.emptybuffer(N*dtype.itemsize)
                    setattr(storage, name, buf)
                else:
                    from runmetal import mem
                    if value is not None:
                        mem.put(value.astype(np.float32), buf.contents())
                return buf
            # if 1:
            with lock:
                # we need to lock this part due to multithreading, it does not crash
                # but gives invalid results (concorrent compute kernels not supported?)
                input_buffers = [getbuf(name, chunk) for name, chunk in zip(self.arguments, chunks)]
                output_buffer = getbuf('vaex_output', N=len(args[0]))
                cqueue, cbuffer = pm.getqueue()
                pm.enqueue_compute(cbuffer, vaex_kernel, input_buffers + [output_buffer])
                pm.enqueue_blit(cbuffer, output_buffer)
            # with lock:
                # we can also move the lock just here, if we devide not to reuse the buffers
                # (uncomment buf = None above)
                pm.start_process(cbuffer)
                pm.wait_process(cbuffer)
                result = pm.buf2numpy(output_buffer, dtype=np.float32)
            return result
        return wrapper

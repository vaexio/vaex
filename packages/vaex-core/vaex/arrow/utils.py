import vaex
import pyarrow as pa
from .convert import trim_offsets

def list_unwrap(ar):
    '''Returns the values in a (nested) list, and a callable that puts it back in the same structure'''
    list_parameters = []
    dtype = vaex.dtype_of(ar)   
    while dtype.is_list:
        list_parameters.append([ar.type, len(ar), ar.buffers(), ar.null_count, ar.offset])
        # flattened.append(ar.type)
        i1 = ar.offsets[0].as_py()
        i2 = ar.offsets[-1].as_py()
        ar = ar.values.slice(i1, i2)
        dtype = dtype.value_type

    def wrapper(new_values):
        if list_parameters:
            ar = None
            for type, length, buffers, null_count, offset in list_parameters[::-1]:
                if ar is None:
                    # buffers = 
                    # assert offset == 0
                    buffers = buffers[:2]
                    buffers = trim_offsets(offset, length, *buffers)
                    offset = 0
                    type = pa.list_(new_values.type)
                    ar = pa.ListArray.from_buffers(type, length, [buffers[0], buffers[1]], null_count, offset, children=[new_values])
                else:
                    ar = pa.ListArray.from_buffers(type, length, [buffers[0], buffers[1]], null_count, offset, children=[ar])
            return ar
        else:
            return new_values
    return ar, wrapper
import vaex
import pyarrow as pa
import pyarrow.compute as pc


def list_unwrap(ar, level=-1):
    '''Returns the values in a (nested) list, and a callable that puts it back in the same structure'''
    from .convert import trim_offsets
    list_parameters = []
    dtype = vaex.dtype_of(ar)
    array_levels = [ar]
    while dtype.is_list:
        list_parameters.append([ar.type, len(ar), ar.buffers(), ar.null_count, ar.offset])
        # flattened.append(ar.type)
        i1 = ar.offsets[0].as_py()
        i2 = ar.offsets[-1].as_py()
        ar = ar.values.slice(i1, i2)
        array_levels.append(ar)
        dtype = dtype.value_type

    if level == -1:
        ar = array_levels[-1]
    else:
        ar = array_levels[level]
        list_parameters = list_parameters[:level+1]

    def wrapper(new_values):
        if list_parameters and list_parameters:
            ar = None
            for type, length, buffers, null_count, offset in list_parameters[::-1]:
                if ar is None:
                    # buffers = 
                    # assert offset == 0
                    buffers = buffers[:2]
                    buffers = trim_offsets(offset, length, *buffers)
                    offset = 0
                    new_values = vaex.array_types.to_arrow(new_values)
                    type = pa.list_(new_values.type)
                    ar = pa.ListArray.from_buffers(type, length, [buffers[0], buffers[1]], null_count, offset, children=[new_values])
                else:
                    ar = pa.ListArray.from_buffers(type, length, [buffers[0], buffers[1]], null_count, offset, children=[ar])
            return ar
        else:
            return new_values
    return ar, wrapper


def combine_missing(a, b):
    # return a copy of a with missing values of a and b combined
    if a.null_count > 0 or b.null_count > 0:
        a, b = vaex.arrow.convert.align(a, b)
        if isinstance(a, pa.ChunkedArray):
            # divide and conquer
            assert isinstance(b, pa.ChunkedArray)
            assert len(a.chunks) == len(b.chunks)
            return pa.chunked_array([combine_missing(ca, cb) for ca, cb in zip(a.chunks, b.chunks)])
        if a.offset != 0:
            a = vaex.arrow.convert.trim_buffers_ipc(a)
        if b.offset != 0:
            b = vaex.arrow.convert.trim_buffers_ipc(b)
        assert a.offset == 0
        assert b.offset == 0
        # not optimal
        nulls = pc.invert(pc.or_(a.is_null(), b.is_null()))
        assert nulls.offset == 0
        nulls_buffer = nulls.buffers()[1]
        # this is not the case: no reason why it should be (TODO: open arrow issue)
        # assert nulls.buffers()[0] is None
        buffers = a.buffers()
        return pa.Array.from_buffers(a.type, len(a), [nulls_buffer] + buffers[1:])
    else:
        return a

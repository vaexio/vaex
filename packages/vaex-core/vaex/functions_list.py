from .functions import register_function
import pyarrow as pa

@register_function(scope='list')
def get(ar, index, default=None):
    pylist = ar.to_pylist()
    return pa.array([k[index] if len(k) > index else default for k in pylist], type=ar.type.value_type)

@register_function(scope='list')
def slice(ar, slice):
    pylist = ar.to_pylist()
    return pa.array([k[slice] for k in pylist], type=ar.type.value_type)
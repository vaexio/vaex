import pyarrow as pa
from .utils import list_unwrap

def test_list_wrap_basics():
    data = ['aap', 'noot', None], None, [], ['aap', 'noot', 'mies']
    string_list = pa.array(data)
    values, wrapper = list_unwrap(string_list)
    assert values.tolist() == string_list.flatten().tolist()

    data_reversed = [['paa', 'toon', None], None, [], ['paa', 'toon', 'seim']]
    reversed = [k[::-1] if k else None for k in values.tolist()]
    assert wrapper(pa.array(reversed)).tolist() == data_reversed


def test_list_wrap_slice():
    data = ['aap', 'noot', None], None, [], ['aap', 'noot', 'mies']
    string_list = pa.array(data)
    values, wrapper = list_unwrap(string_list.slice(1))
    assert values.tolist() == string_list.slice(1).flatten().tolist()

    data_reversed = [None, [], ['paa', 'toon', 'seim']]
    reversed = [k[::-1] if k else None for k in values.tolist()]
    assert wrapper(pa.array(reversed)).tolist() == data_reversed



def test_list_list_wrap():
    data = [['aap', 'noot', None], None, [], ['aap', 'noot', 'mies']], [], None
    string_list_list = pa.array(data)
    values, wrapper = list_unwrap(string_list_list)
    assert values.tolist() == string_list_list.flatten().flatten().tolist()    
    assert wrapper(values).tolist() == string_list_list.tolist()


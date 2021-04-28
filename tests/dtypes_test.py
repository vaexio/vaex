from common import *
from vaex.datatype import DataType

def test_dtype_basics(df):
    df['new_virtual_column'] = df.x + 1
    for name in df.get_column_names():
        if df.is_string(name):
            assert df[name].to_numpy().dtype.kind in 'OSU'
        else:
            assert vaex.array_types.same_type(DataType(vaex.array_types.data_type(df[name].values)), df.data_type(df[name]))


def test_dtypes(df_local):
    df = df_local
    assert [df.dtypes[name] for name in df.get_column_names()] == [df[name].data_type() for name in df.get_column_names()]


def test_dtype_arrow():
    l = pa.array([[1,2], [2,3,4]])
    df = vaex.from_arrays(l=l)
    assert df.data_type(df.l) == pa.list_(l.type.value_type)


def test_dtype_str():
    df = vaex.from_arrays(x=["foo", "bars"], y=[1,2])
    assert df.data_type(df.x) == pa.string()
    assert df.data_type(df.x, array_type='arrow') == pa.string()
    df['s'] = df.y.apply(lambda x: str(x))
    assert df.data_type(df.x) == pa.string()
    assert df.data_type(df.s) == pa.string()
    assert df.data_type(df.x, array_type='arrow') == pa.string()
    assert df.data_type(df.s, array_type='arrow') == pa.string()

    assert df.data_type(df.x.as_arrow(), array_type=None) == pa.string()
    assert df.data_type(df.x.as_arrow(), array_type='arrow') == pa.string()
    assert df.data_type(df.x.as_arrow(), array_type='numpy') == object

    n = np.array(['aap', 'noot'])
    assert vaex.from_arrays(n=n).n.dtype == pa.string()

    n = np.array([np.nan, 'aap', 'noot'], dtype=object)
    df = vaex.from_arrays(n=n)
    assert df.n.dtype == pa.string()
    assert df.copy().n.dtype == pa.string()

    n = np.array([None, 'aap', 'noot'])
    df = vaex.from_arrays(n=n)
    assert df.n.dtype == pa.string()
    assert df.copy().n.dtype == pa.string()


def test_dtype_str_invalid_identifier():
    df = vaex.from_dict({'#': ['foo']})
    assert df.data_type('#') == 'string'
    assert df.data_type('#', array_type='numpy') == 'object'
    assert df.data_type('#', array_type='numpy-arrow') == 'string'
    assert df['#'].dtype == 'string'


def test_dtype_str_virtual_column():
    df = vaex.from_dict({'s': ['foo']})
    df['v'] = df.s.str.lower()
    assert df.data_type('v') == 'string'
    assert df.data_type('v', array_type='numpy') == 'object'
    assert df.data_type('v', array_type='numpy-arrow') == 'string'
    assert df['v'].dtype == 'string'


def test_dtype_nested():
    data = ['aap', 'noot', None], ['app', 'noot', 'mies']
    df = vaex.from_arrays(s=pa.array(data))
    assert df.s.dtype == pa.list_(pa.string())
    assert df.s.data_type(axis=0) == pa.list_(pa.string())
    assert df.s.data_type(axis=-2) == pa.list_(pa.string())
    assert df.s.data_type(axis=1) == pa.string()
    assert df.s.data_type(axis=-1) == pa.string()

    data = [['aap', 'noot', None], ['app', 'noot', 'mies']], [], None
    df = vaex.from_arrays(s=pa.array(data))
    assert df.s.dtype == pa.list_(pa.list_(pa.string()))
    assert df.s.data_type(axis=-3) == pa.list_(pa.list_(pa.string()))
    assert df.s.data_type(axis=-2) == pa.list_(pa.string())
    assert df.s.data_type(axis=-1) == pa.string()

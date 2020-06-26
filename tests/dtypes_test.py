from common import *


def test_dtype_basics(df):
    df['new_virtual_column'] = df.x + 1
    for name in df.get_column_names():
        if df.is_string(name):
            assert df[name].to_numpy().dtype.kind in 'OSU'
        else:
            assert df[name].values.dtype == df.data_type(df[name])


def test_dtypes(df_local):
    df = df_local
    assert [df.dtypes[name] for name in df.get_column_names()] == [df[name].data_type() for name in df.get_column_names()]


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
    assert 'n' in df._dtypes_override

    n = np.array([None, 'aap', 'noot'])
    df = vaex.from_arrays(n=n)
    assert df.n.dtype == pa.string()
    assert df.copy().n.dtype == pa.string()
    assert 'n' in df._dtypes_override

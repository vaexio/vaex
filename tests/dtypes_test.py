from common import *
from vaex.column import str_type



def test_dtype_basics(df):
    df['new_virtual_column'] = df.x + 1
    for name in df.column_names:
        if df.dtype(name) == str_type:
            assert df[name].values.dtype.kind in 'OSU'
        else:
            assert df[name].values.dtype == df.dtype(df[name])


def test_dtypes(ds_local):
    ds = ds_local
    assert [ds.dtypes[name] for name in ds.column_names] == [ds[name].dtype for name in ds.column_names]


def test_dtype_str():
    df = vaex.from_arrays(x=["foo", "bars"], y=[1,2])
    assert df.dtype(df.x) == str_type
    df['s'] = df.y.apply(lambda x: str(x))
    assert df.dtype(df.x) == str_type
    assert df.dtype(df.s) == str_type

    n = np.array(['aap', 'noot'])
    assert vaex.from_arrays(n=n).n.dtype == str_type

    n = np.array([np.nan, 'aap', 'noot'], dtype=object)
    df = vaex.from_arrays(n=n)
    assert df.n.dtype == str_type
    assert df.copy().n.dtype == str_type
    assert 'n' in df._dtypes_override

    n = np.array([None, 'aap', 'noot'])
    df = vaex.from_arrays(n=n)
    assert df.n.dtype == str_type
    assert df.copy().n.dtype == str_type
    assert 'n' in df._dtypes_override

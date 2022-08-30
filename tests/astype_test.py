from common import *
import collections
import numpy as np

def test_astype(ds_local):
    ds = ds_local
    ds_original = ds.copy()
    #ds.columns['x'] = (ds.columns['x']*1).copy()  # convert non non-big endian for now
    ds['x'] = ds['x'].astype('f4')

    assert ds.x.dtype == 'float32'
    assert ds.x.tolist() == ds_original.x.as_numpy().evaluate().astype(np.float32).tolist()


def test_astype_str():
    df = vaex.from_arrays(x=['10,010', '-50,0', '11,111'])

    df['x'] = df['x'].str.replace(',', '').evaluate()
    df['x'] = (df['x'].astype('float')).astype('int64').evaluate()

    assert df.x.dtype == int


@pytest.fixture
def nullable_ints(df_factory):
    return df_factory(x=[1, 2, None])


@pytest.mark.parametrize('dtype', [str, 'str', 'string', pa.string()])
def test_astype_to_str(nullable_ints, dtype):
    assert nullable_ints.x.astype(dtype).tolist() == ['1', '2', None]


@pytest.mark.parametrize(
    'dtype',
    [
        'int',
        'i1',
        'u1',
        'int8',
        'uint8',
        pa.int8(),
        pa.uint8(),
        'int16',
        'int32',
        'int64',
    ]
)
def test_astype_integral(nullable_ints, dtype):
    assert nullable_ints.x.astype(dtype).tolist() == [1, 2, None]


@pytest.mark.parametrize(
    'dtype',
    [
        'float',
        'f4',
        'f8',
        'float32',
        'float64',
        pa.float32(),
        pa.float64(),
    ]
)
def test_astype_floating(nullable_ints, dtype):
    assert nullable_ints.x.astype(dtype).tolist() == [1., 2., None]

@pytest.mark.parametrize('dtype', [int, float, bool, np.int8, np.float32])
def test_astype_numeric_fails(nullable_ints, dtype):
    with pytest.raises((ValueError, TypeError)):
        nullable_ints.x.astype(dtype).tolist()



def test_astype_dtype():
    df = vaex.from_arrays(x=[0, 1])
    assert df.x.astype(str).data_type() in [pa.string(), pa.large_string()]
    df = vaex.from_arrays(x=[np.nan, 1])
    # assert df.x.astype(str).dtype == vaex.column.str_type
    assert df.x.astype(str).data_type() in [pa.string(), pa.large_string()]


def test_astype_empty(df_factory):
    df = df_factory(x=[1, 2, 3])
    df = df[df.x<0]
    assert len(df.x.as_numpy().values) == 0


def test_astype_timedelta(df_factory):
    x = [23423, -242, 34656]
    x_result = np.array([23423,  -242, 34656], dtype='timedelta64[s]')
    df = df_factory(x=x)
    df['x_expected'] = df.x.astype('timedelta64[s]')
    assert x_result.tolist() == df.x_expected.tolist()


def test_astype_str_to_datetime(df_factory):
    x = ['2020-05', '2021-10', '2022-01']
    y = ['2020', '2021', '2022']
    x_validation = np.array(x, dtype='datetime64[M]')
    y_validation = np.array(y, dtype='datetime64[Y]')
    df = df_factory(x=x, y=y)
    df['x_dt'] = df.x.astype('datetime64[M]')
    df['y_dt'] = df.y.astype('datetime64[Y]')
    assert all(df.x_dt.values == x_validation)
    assert all(df.y_dt.values == y_validation)

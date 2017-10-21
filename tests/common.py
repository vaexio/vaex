import pytest
import vaex
import numpy as np

@pytest.fixture()
def ds_filtered():
    return create_filtered()

@pytest.fixture()
def ds_half():
    ds = create_base_ds()
    ds.set_active_range(0, 10)
    return ds

@pytest.fixture()
def ds_trimmed():
    ds = create_base_ds()
    ds.set_active_range(0, 10)
    return ds.trim()

@pytest.fixture(params=[ds_filtered, ds_half, ds_trimmed], ids=['ds_filtered', 'ds_half', 'ds_trimmed'])
def ds(request):
    return request.param()

@pytest.fixture(params=[ds_half, ds_trimmed], ids=['ds_half', 'ds_trimmed'])
def ds_no_filter(request):
    return request.param()

def create_filtered():
    ds = create_base_ds()
    ds.select('x < 10', name=vaex.dataset.FILTER_SELECTION_NAME)
    return ds

def create_base_ds():
    dataset = vaex.dataset.DatasetArrays("dataset")
    x = x = np.arange(40, dtype=">f8").reshape((-1,20)).T.copy()[:,0]
    y = y = x ** 2
    ints = np.arange(20, dtype="i8")
    ints[0] = 2**62+1
    ints[1] = -2**62+1
    ints[2] = -2**62-1
    ints[0+10] = 2**62+1
    ints[1+10] = -2**62+1
    ints[2+10] = -2**62-1
    dataset.add_column("x", x)
    dataset.add_column("y", y)
    m = x.copy()
    ma_value = 77777
    m[-1+10] = ma_value
    m[-1+20] = ma_value
    m = np.ma.array(m, mask=m==ma_value)

    n = x.copy()
    n[-2+10] = np.nan
    n[-2+20] = np.nan

    nm = x.copy()
    nm[-2+10] = np.nan
    nm[-2+20] = np.nan
    nm[-1+10] = ma_value
    nm[-1+20] = ma_value
    nm = np.ma.array(nm, mask=nm==ma_value)

    mi = mi = np.ma.array(m.data.astype(np.int64), mask=m.data==ma_value, fill_value=88888)
    dataset.add_column("m", m)
    dataset.add_column('n', n)
    dataset.add_column('nm', nm)
    dataset.add_column("mi", mi)
    dataset.add_column("ints", ints)


    name = np.array(list(map(lambda x: str(x) + "bla" + ('_' * int(x)), x)), dtype='S') #, dtype=np.string_)
    dataset.add_column("name", np.array(name))
    return dataset

dsf = create_filtered()
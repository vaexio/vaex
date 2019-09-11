from common import *
import dask
import dask.array as da

@pytest.fixture
def df():
    x = np.arange(10, dtype=np.float64)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    df = df[['x', 'y']] # make sure we are in this order
    return df

def test_dask_array(df):
    X = np.array(df)
    Xd = df.to_dask_array(chunks=(5,1)).compute()
    assert Xd.tolist() == X.tolist()

    Xd = (df.to_dask_array(chunks=(5,1))**2).compute()
    assert Xd.tolist() == (X**2).tolist()

def test_dask_sin(df):
    X = np.array(df)
    Xdv = df.to_dask_array()
    sines = np.sin(Xdv).compute()
    assert sines[:,0].tolist() == np.sin(df.x).tolist()
    assert sines[:,1].tolist() == np.sin(df.y).tolist()


def test_dask_svd(df):
    Xd = df.to_dask_array(chunks=(10,2))
    values = da.linalg.svd(Xd)
    values = dask.compute(*values)
    values = [k.tolist() for k in values]

    # X = np.array(df)
    # Xd = da.from_array(X)
    # # Xdvaex = da.linalg.qr()
    # Av, bv = np.linalg.svd(df)
    # A, b = dask.compute(*np.linalg.qr(Xd))
    # # Av, bv = dask.compute(*np.linalg.qr(Xdvaex), scheduler='single-threaded')
    # assert b.tolist() == bv.tolist()
    # assert A.tolist() == bv.tolist()

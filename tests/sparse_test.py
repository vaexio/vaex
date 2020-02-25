from common import *
from scipy.sparse import csc_matrix, csr_matrix


def test_sparse_basics():
    x = np.arange(3)
    s = csr_matrix([[0, 1], [1, 0], [2, 3]])
    ds = vaex.from_arrays(x=x)
    ds.add_columns(['s1', 's2'], s)
    assert ds.s1.tolist() == [0, 1, 2]
    assert ds.s2.tolist() == [1, 0, 3]
    assert "error" not in repr(ds)


def test_sparse_export(tmpdir):
    path = str(tmpdir.join('test.hdf5'))
    x = np.arange(3)
    ds = vaex.from_arrays(x=x)
    s = csr_matrix([[0, 1], [1, 0], [2, 3]])
    ds.add_columns(['s1', 's2'], s)

    p = csr_matrix([[4, 0], [0, 0], [9, 10]])
    ds.add_columns(['p1', 'p2'], p)

    ds.export_hdf5(path)

    ds2 = vaex.open(path)
    assert ds2.s1.tolist() == [0, 1, 2]
    assert ds2.s2.tolist() == [1, 0, 3]

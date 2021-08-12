import vaex
import pyarrow as pa

def test_get():
    df = vaex.from_arrays(l=pa.array([[1,2], [2,3,4]]))
    assert df.l.list.get(0).tolist() == [1, 2]
    assert df.l.list.get(1).tolist() == [2, 3]
    assert df.l.list.get(2).tolist() == [None, 4]
    assert df.l.list.get(2, -1).tolist() == [-1, 4]

def test_slice():
    df = vaex.from_arrays(l=pa.array([[1,2], [2,3,4], [3,4,5,6]]))
    assert df.l.list[:1].tolist() == [[1], [2], [4,5,6]]
    assert df.l.list[:2].tolist() == [2, 3]
    assert df.l.list[1:2].tolist() == [None, 4]
    assert df.l.list[2:].tolist() == [-1, 4]


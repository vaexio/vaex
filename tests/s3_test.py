import vaex

def test_s3():
    df = vaex.open('s3://vaex/testing/xy.hdf5?cache=false')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [2, 3]

    df = vaex.open('s3://vaex/testing/xy.hdf5?cache=true')
    assert df.x.tolist() == [1, 2]
    assert df.y.tolist() == [2, 3]

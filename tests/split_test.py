from common import *

def test_split(ds_local):
    ds = ds_local
    ds1, ds2 = ds.split(0.5)
    assert len(ds1) == len(ds2)
    assert ds1.x.tolist() + ds2.x.tolist() == ds.x.tolist()

    ds1, ds2 = ds.split_random(0.5, random_state=42)
    assert ds1.x.tolist() + ds2.x.tolist() != ds.x.tolist()
    assert list(sorted(ds1.x.tolist() + ds2.x.tolist())) == ds.x.tolist()

    # 1+2+3+4=10
    ds1, ds2, ds3, ds4 = ds.split([1,2,3,4])
    assert len(ds1) == 1
    assert len(ds2) == 2
    assert len(ds3) == 3
    assert len(ds4) == 4
    assert ds1.x.tolist() + ds2.x.tolist() \
         + ds3.x.tolist() + ds4.x.tolist()  == ds.x.tolist()

    ds1, ds2, ds3, ds4 = ds.split_random([1,2,3,4], random_state=42)
    assert len(ds1) == 1
    assert len(ds2) == 2
    assert len(ds3) == 3
    assert len(ds4) == 4
    assert list(sorted(ds1.x.tolist() + ds2.x.tolist() \
                     + ds3.x.tolist() + ds4.x.tolist()))  == ds.x.tolist()


def test_split_n(df_local):
     df = df_local
     assert len(df.split(3)) == 3
     assert [len(k) for k in df.split(3)] == [4, 4, 2]
     assert vaex.concat(df.split(3)).x.tolist() == df.x.tolist()
     assert len(df.split(10)) == 10
     assert vaex.concat(df.split(10)).x.tolist() == df.x.tolist()
     assert len(df.split(11)) == 10
     assert len(df.split(1)) == 1
     assert len(df.split(1)[0]) == 10

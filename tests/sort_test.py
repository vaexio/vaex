from common import *


def test_sort(ds_local):
    ds = ds_local
    print(ds, ds_trimmed)
    x = np.arange(10).tolist()
    dss = ds.sample(frac=1, random_state=42)
    assert dss.x.evaluate().tolist() != x  # make sure it is not sorted

    ds_sorted = dss.sort('x')
    assert ds_sorted.x.evaluate().tolist() == x

    ds_sorted = dss.sort('-x')
    assert ds_sorted.x.evaluate().tolist() == x[::-1]

    ds_sorted = dss.sort('-x', ascending=False)
    assert ds_sorted.x.evaluate().tolist() == x

    ds_sorted = dss.sort('x', ascending=False)
    assert ds_sorted.x.evaluate().tolist() == x[::-1]


def test_sort_filtered():
    x = [2, 3, 1, 5, 7, 6]
    df = vaex.from_arrays(x=x)
    df_sel_sorted = df[df.x > 4].sort(by='x')
    assert df_sel_sorted.x.tolist() == [5, 6, 7]

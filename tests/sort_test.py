from common import *


def test_sort(ds_local):
    ds = ds_local
    print(ds, ds_trimmed)
    x = np.arange(10).tolist()
    dss = ds.sample(frac=1, random_state=42)
    assert dss.x.tolist() != x  # make sure it is not sorted

    ds_sorted = dss.sort('x')
    assert ds_sorted.x.tolist() == x

    ds_sorted = dss.sort('-x')
    assert ds_sorted.x.tolist() == x[::-1]

    ds_sorted = dss.sort('-x', ascending=False)
    assert ds_sorted.x.tolist() == x

    ds_sorted = dss.sort('x', ascending=False)
    assert ds_sorted.x.tolist() == x[::-1]


def test_sort_filtered():
    x = [2, 3, 1, 5, 7, 6]
    df = vaex.from_arrays(x=x)
    df_sel_sorted = df[df.x > 4].sort(by='x')
    assert df_sel_sorted.x.tolist() == [5, 6, 7]

def test_sort_multikey():
    x = np.array([5, 3, 1, 1, 5])
    y = np.array([0, 3, 4, 2, 1])
    z = np.array(['dog', 'cat', 'cat', 'dog', 'mouse'])
    df = vaex.from_arrays(x=x, y=y, z=z)

    # Case 1: numeric keys
    df_sorted_1 = df.sort(by=['x', 'y'])
    assert df_sorted_1.x.tolist() == [1, 1, 3, 5, 5]
    assert df_sorted_1.y.tolist() == [2, 4, 3, 0, 1]
    assert df_sorted_1.z.tolist() == ['dog', 'cat', 'cat', 'dog', 'mouse']

    # Case 2: str key followed by numeric key
    df_sorted_2 = df.sort(by=['z', 'x'])
    assert df_sorted_2.x.tolist() ==  [1, 3, 1, 5, 5]
    assert df_sorted_2.y.tolist() ==  [4, 3, 2, 0, 1]
    assert df_sorted_2.z.tolist() == ['cat', 'cat', 'dog', 'dog', 'mouse']

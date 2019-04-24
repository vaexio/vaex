from common import *

# def test_count_edges():
#     ds = vaex.from_arrays(x=[-2, -1, 0, 1, 2, 3, np.nan])
#     # 1 missing, 2 to the left, 1 in the range, two to the right
#     assert ds.count(ds.x, binby=ds.x, limits=[0.5, 1.5], shape=1, edges=False).tolist() == [1]
#     assert ds.count(binby=ds.x, limits=[0.5, 1.5], shape=1, edges=True).tolist() == [1, 3, 1, 2]
#     # if the value itself is nan, it should not count it
#     assert ds.count(ds.x, binby=ds.x, limits=[0.5, 1.5], shape=1, edges=True).tolist() == [0, 3, 1, 2]

#     x = np.array([-2, -1, 0, 1, 2, 3, 4])
#     x = np.ma.array(x, mask=x==4)
#     ds = vaex.from_arrays(x=x)
#     # same, but now with a masked value
#     assert ds.count(binby=ds.x, limits=[0.5, 1.5], shape=1, edges=True).tolist() == [1, 3, 1, 2]
#     # assert ds.count(ds.x, limits=[0.5, 1.5], shape=2, edges=True).tolist() == [1, 3, 1, 2]

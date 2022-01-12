import numpy as np

def test_hashmap_unique_basics(df_local):
    df = df_local
    xmap = df._hash_map_unique('x')
    assert set(xmap.keys().tolist()) == set(range(10))

    assert xmap.sorted().keys().tolist() == list(range(10))

def test_hashmap_unique_missing(df_local):
    df = df_local
    mmap = df._hash_map_unique('m')
    expected = list(range(10))
    expected[9] = None
    result = mmap.keys().tolist()
    assert set(result) == set(expected)    

    for limit in range(1, 10):
        result_limit = mmap.limit(limit).keys()
        assert result_limit.tolist() == result[:limit]


    result_sorted = mmap.sorted().keys().tolist()
    assert set(result_sorted) == set(expected)


def test_hashmap_unique_nan(df_local):
    df = df_local
    nmmap = df._hash_map_unique('nm')
    assert np.isnan(nmmap.keys()).sum() == 1

    expected = list(range(10))
    expected[7] = None
    # we can't compare nan, so use 999 as proxy
    expected[6] = 999

    result = nmmap.keys()
    result = np.ma.where(np.isnan(result), 999, result).tolist()
    assert set(result) == set(expected)

    for limit in range(1, 10):
        result_limit = nmmap.limit(limit).keys()
        result_limit = np.ma.where(np.isnan(result_limit), 999, result_limit)
        assert result_limit.tolist() == result[:limit]

    result_sorted = nmmap.sorted().keys()
    expected_sorted = [0, 1, 2, 3, 4, 5, 8, 9, None, 999]
    result_sorted = np.ma.where(np.isnan(result_sorted), 999, result_sorted).tolist()
    assert set(result_sorted) == set(expected_sorted)
    assert np.isnan(nmmap.sorted().keys()).sum() == 1


def test_hashmap_unique_strings(df_factory):
    expected = ['aap', 'noot', 'mies', None, 'kees']
    df = df_factory(s=expected)
    mmap = df._hash_map_unique('s')
    result = mmap.keys().tolist()
    assert set(result) == set(expected)    

    for limit in range(1, 10):
        result_limit = mmap.limit(limit).keys()
        assert result_limit.tolist() == result[:limit]

    result_sorted = mmap.sorted().keys().tolist()
    assert set(result_sorted) == set(expected)

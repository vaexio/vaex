import pytest
import numpy as np
import vaex.cache
from unittest.mock import MagicMock, call


def passes(df):
    return df.executor.passes if df.is_local() else df.executor.remote_calls


def reset(df):
    if df.is_local():
        df.executor.passes = 0
    else:
        df.executor.remote_calls = 0


def test_memory():
    with vaex.cache.off():
        assert vaex.cache.cache is None
        with vaex.cache.memory_infinite():
            assert isinstance(vaex.cache.cache, dict)
        assert vaex.cache.cache is None
        vaex.cache.memory_infinite()
        assert isinstance(vaex.cache.cache, dict)


def test_on():
    with vaex.cache.off():
        assert vaex.cache.cache is None
        with vaex.cache.on():
            assert isinstance(vaex.cache.cache, dict)
        assert vaex.cache.cache is None
        vaex.cache.on()
        assert isinstance(vaex.cache.cache, dict)
        vaex.cache.off()
        assert vaex.cache.cache is None

    with vaex.cache.on("memory_infinite,disk"):
        import diskcache
        assert isinstance(vaex.cache.cache, vaex.cache.MultiLevelCache)
        assert isinstance(vaex.cache.cache.maps[0], dict)
        assert isinstance(vaex.cache.cache.maps[1], diskcache.Cache)
    assert not vaex.cache.is_on()



def test_cached_result(df_local):
    with vaex.cache.memory_infinite(clear=True):
        assert vaex.cache.is_on()
        df = df_local._future() # v4 does a pass when the aggregation depends on a column
        reset(df)
        sum0 = df.sum('x', delay=True)
        # even if we add a second aggregation which is merged, we should find the sum
        # in cache the next time
        mean0 = df.mean('x', delay=True)
        df.execute()
        assert passes(df) == 1
        sum0 = sum0.get()
        
        # now it should be cached, ...
        reset(df)
        sum0b = df.sum('x')
        assert sum0 == sum0b
        # so no extra passes
        assert passes(df) == 0

        reset(df)
        df = df[df.x < 4]
        total = 1 + 2 + 3

        # this should be a new result
        sum1_filtered = df.sum('x')
        assert passes(df) == 1
        assert sum1_filtered == total

        reset(df)
        # now it should be cached
        sum1b_filtered = df.sum('x')
        assert passes(df) == 0
        assert sum1b_filtered == total

        # and it should not care if we add a virtual column we do not use
        df['foo'] = df.x * 2
        sum1b_filtered = df.sum('x')
        assert passes(df) == 0
        assert sum1b_filtered == total


def test_cached_result_array(df_local):
    with vaex.cache.memory_infinite(clear=True):
        assert vaex.cache.is_on()
        df = df_local._future() # v4 does a pass when the aggregation depends on a column
        reset(df)
        sum0 = df.sum('x', binby='y', delay=True)
        sum0b = df.sum('x', binby='y', delay=True)
        df.execute()
        assert passes(df) == 2, 'one pass for min/max, one for the aggregation'

        reset(df)
        sum0 = df.sum('x', binby='y')
        assert passes(df) == 0


def test_cache_length_without_messing_up_filter_mask(df_local):
    df = df_local
    with vaex.cache.memory_infinite(clear=True):
        dff = df[df.x < 4]
        passes0 = passes(df)
        len(dff)
        assert passes(df) == passes0 + 1
        # create a new df, that should re-use the results of the filter
        dff = df[df.x < 4]
        # this should not trigger a computation (due to cache)
        len(dff)
        assert passes(df) == passes0 + 1
        # slicing needs the mask, so it should trigger
        dffs = dff[1:2]
        assert passes(df) == passes0 + 2


def test_cache_set():
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        passes0 = passes(df)
        df._set('x')
        assert passes(df) == passes0 + 1
        # now should use the cache
        df._set('x')
        assert passes(df) == passes0 + 1


def test_nunique():
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        vaex.cache._cache_hit = 0
        vaex.cache._cache_miss = 0
        df.x.nunique()
        # twice in the exector for the set, 1 time in nunique
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 0
        df.x.nunique()
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 1


@pytest.mark.parametrize("copy", [True, False])
def test_cache_groupby(copy):
    df = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    df2 = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    fp = df.fingerprint()  # we also want to be sure we don't modify the fingerprint
    with vaex.cache.memory_infinite(clear=True):
        passes0 = passes(df)

        df.groupby('x', agg='count', copy=copy)
        # we do two passes, one for the set, and 1 for the aggregation
        assert passes(df) == passes0 + 2
        if copy:
            assert df.fingerprint() == fp

        df.groupby('y', agg='count', copy=copy)
        # we do two passes, one for the set, and 1 for the aggregation
        assert passes(df) == passes0 + 4
        if copy:
            assert df.fingerprint() == fp

        vaex.execution.logger.debug("HERE IT GOES")
        df2.groupby('y', agg='count', copy=copy)
        # different dataframe, same data, no extra passes
        assert passes(df) == passes0 + 4
        if copy:
            assert df.fingerprint() == fp

        # now should use the cache
        df.groupby('x', agg='count', copy=copy)
        assert passes(df) == passes0 + 4
        df.groupby('y', agg='count', copy=copy)
        assert passes(df) == passes0 + 4
        if copy:
            assert df.fingerprint() == fp

        # 1 time for combining the two sets, 1 for finding the labels (smaller dataframe), 1 pass for aggregation
        df.groupby(['x', 'y'], agg='count', copy=copy)
        assert passes(df) == passes0 + 7
        if copy:
            assert df.fingerprint() == fp

        # but that should be reused the second time, also the evaluation of the labels
        df.groupby(['x', 'y'], agg='count', copy=copy)
        assert passes(df) == passes0 + 7
        if copy:
            assert df.fingerprint() == fp

        # different order triggers a new pass(combining the sets in a different way)
        # and the aggregation phase, which includes the labeling
        df.groupby(['y', 'x'], agg='count', copy=copy)
        assert passes(df) == passes0 + 7 + 3
        if copy:
            assert df.fingerprint() == fp


def test_cache_selections():
    df = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    df2 = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    fp = df.fingerprint()  # we also want to be sure we don't modify the fingerprint
    with vaex.cache.memory_infinite(clear=True):
        df.executor.passes = 0
        assert df.x.sum(selection=df.y=='b') == 1
        assert passes(df) == 1

        # same data, different dataframe, no extra pass
        assert df2.x.sum(selection=df2.y=='b') == 1
        assert passes(df) == 1

        # named selections
        df.executor.passes = 0
        df.select(df.y=='c', name="a")
        assert df.x.sum(selection="a") == 2
        assert passes(df) == 1

        df2.select(df2.y=='c', name="a")
        assert df2.x.sum(selection="a") == 2
        assert passes(df2) == 1

        df['z'] = df.x * 2
        df2['z'] = df2.x * 2

        # named selection referring to a virtual column
        df.executor.passes = 0
        df.select(df.z==4, name="a")
        assert df.x.sum(selection="a") == 4
        assert passes(df) == 1

        df2.select(df2.z==4, name="a")
        assert df2.x.sum(selection="a") == 4
        assert passes(df2) == 1

        df['z'] = df.x * 3
        df2['z'] = df2.x * 1  # different virtual column with same name
        # named selection referring to a virtual column, but different expressions
        df.executor.passes = 0
        df.select(df.z==3, name="a")
        assert df.x.sum(selection="a") == 1
        assert passes(df) == 1

        # we should not get the cached version now
        df2.select(df2.z==3, name="a")
        assert df2.x.sum(selection="a") == 0
        assert passes(df2) == 2


def test_multi_level_cache():
    l1 = {}
    l2 = {}
    cache = vaex.cache.MultiLevelCache(l1, l2)
    with pytest.raises(KeyError):
        value = cache['key1']
    assert l1 == {}
    assert l2 == {}
    # setting should fill all caches
    cache['key1'] = 1
    assert l1 == {'key1': 1}
    assert l2 == {'key1': 1}
    assert cache['key1'] == 1
    del l1['key1']
    assert l1 == {}
    assert l2 == {'key1': 1}
    # reading should fill l1 as well
    assert cache['key1'] == 1
    assert l1 == {'key1': 1}
    assert l2 == {'key1': 1}


def test_memoize():
    f1 = f1_mock = MagicMock()
    f1b = f1b_mock = MagicMock()
    f2 = f2_mock = MagicMock()

    f1 = vaex.cache._memoize(f1, key_function=lambda: 'same')
    f1b = vaex.cache._memoize(f1b, key_function=lambda: 'same')
    f2 = vaex.cache._memoize(f2, key_function=lambda: 'different')

    with vaex.cache.memory_infinite(clear=True):
        f1()
        f1_mock.assert_called_once()
        f1b_mock.assert_not_called()
        f2_mock.assert_not_called()
        f1b()
        f1_mock.assert_called_once()
        f1b_mock.assert_not_called()
        f2_mock.assert_not_called()
        f2()
        f1_mock.assert_called_once()
        f1b_mock.assert_not_called()
        f2_mock.assert_called_once()

    with vaex.cache.off():
        f1()
        f1_mock.assert_has_calls([call(), call()])
        f1b_mock.assert_not_called()
        f2_mock.assert_called_once()
        f1b()
        f1_mock.assert_has_calls([call(), call()])
        f1b_mock.assert_called_once()
        f2_mock.assert_called_once()
        f2()
        f1_mock.assert_has_calls([call(), call()])
        f1b_mock.assert_called_once()
        f2_mock.assert_has_calls([call(), call()])


def test_memoize_with_delay():
    # using nunique for this
    df = vaex.from_arrays(x=[0, 1, 2, 2])
    with vaex.cache.memory_infinite(clear=True):
        vaex.cache._cache_hit = 0
        vaex.cache._cache_miss = 0
        value = df.x.nunique(delay=True)
        # twice for the tasks of unique
        assert vaex.cache._cache_miss == 2
        assert vaex.cache._cache_hit == 0
        df.execute()
        assert value.get() == 3
        # 1 time in nunique
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 0

        value = df.x.nunique()
        assert value == 3
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 1


        value = df.x.nunique(delay=True)
        assert value.get() == 3
        assert vaex.cache._cache_miss == 3
        assert vaex.cache._cache_hit == 2

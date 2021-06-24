import numpy as np
import vaex.cache


def passes(df):
    return df.executor.passes if df.is_local() else df.executor.remote_calls


def reset(df):
    if df.is_local():
        df.executor.passes = 0
    else:
        df.executor.remote_calls = 0


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
        passes1b = passes(df)
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


def test_cache_groupby():
    df = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    df2 = vaex.from_arrays(x=[0, 1, 2, 2], y=['a', 'b', 'c', 'd'])
    fp = df.fingerprint()  # we also want to be sure we don't modify the fingerprint
    with vaex.cache.memory_infinite(clear=True):
        passes0 = passes(df)

        df.groupby('x', agg='count')
        # we do two passes, one for the set, and 1 for the aggregation
        assert passes(df) == passes0 + 2
        assert df.fingerprint() == fp

        df.groupby('y', agg='count')
        # we do two passes, one for the set, and 1 for the aggregation
        assert passes(df) == passes0 + 4
        assert df.fingerprint() == fp

        df2.groupby('y', agg='count')
        # different dataframe, same data, no extra passes
        assert passes(df) == passes0 + 4
        assert df.fingerprint() == fp

        # now should use the cache
        df.groupby('x', agg='count')
        assert passes(df) == passes0 + 4
        df.groupby('y', agg='count')
        assert passes(df) == passes0 + 4
        assert df.fingerprint() == fp

        # 1 time for combining the two sets, 1 for finding the labels (smaller dataframe), 1 pass for aggregation
        df.groupby(['x', 'y'], agg='count')
        assert passes(df) == passes0 + 7
        assert df.fingerprint() == fp

        # but that should be reused the second time, just 1 pass for the evaluations of the labels
        df.groupby(['x', 'y'], agg='count')
        assert passes(df) == passes0 + 7 + 1
        assert df.fingerprint() == fp

        # different order triggers a new pass(combining the sets in a different way)
        # and the aggregation phase, which includes the labeling
        df.groupby(['y', 'x'], agg='count')
        assert passes(df) == passes0 + 7 + 1 + 3
        assert df.fingerprint() == fp


import numpy as np
import vaex.cache

def passes(df):
    return df.executor.passes if df.is_local() else df.executor.remote_calls

def test_cached_result(df_local):
    with vaex.cache.infinite():
        assert vaex.cache.is_on()
        df = df_local
        len(df)  # trigger a first pass (filtering) TODO: why is this needed?
        passes0 = passes(df)
        sum0 = df.sum('x')
        passes1 = passes(df)
        assert passes1 == passes0 + 1
        
        # now it should be cached, ...
        sum0b = df.sum('x')
        assert sum0 == sum0b
        # so no extra passes
        passes1b = passes(df)
        assert passes1 == passes1b


        df = df[df.x < 4]
        len(df)  # trigger a filter pass
        passes0 = passes(df)
        total = 1 + 2 + 3        

        # this should be a new result
        sum1_filtered = df.sum('x')
        passes1 = passes(df)
        assert passes1 == passes0 + 1
        assert sum1_filtered == total

        # now it should be cached
        sum1b_filtered = df.sum('x')
        passes1b = passes(df)
        assert passes1b == passes1
        assert sum1b_filtered == sum1_filtered


def test_cache_length_without_messing_up_filter_mask(df_local):
    df = df_local
    with vaex.cache.infinite():
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


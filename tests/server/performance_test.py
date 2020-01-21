import pytest
import contextlib


@contextlib.contextmanager
def small_buffer(ds, size=3):
    if ds.is_local():
        previous = ds.executor.buffer_size
        ds.executor.buffer_size = size
        ds._invalidate_selection_cache()
        try:
            yield
        finally:
            ds.executor.buffer_size = previous
    else:
        yield # for remote datasets we don't support this ... or should we?


def test_delayed(df_server, df_remote, webserver, client):
    xmin = df_server.x.min()
    xmax = df_server.x.max()
    remote_calls = df_remote.executor.remote_calls
    # import pdb; pdb.set_trace()
    assert df_remote.x.min() == xmin
    assert df_remote.executor.remote_calls == remote_calls + 1

    # off, top passes, bottom does not
    remote_calls = df_remote.executor.remote_calls
    df_remote.x.min()
    df_remote.x.max()
    assert df_remote.executor.remote_calls == remote_calls + 2

    remote_calls = df_remote.executor.remote_calls
    vmin = df_remote.x.min(delay=True)
    vmax = df_remote.x.max(delay=True)
    assert df_remote.executor.remote_calls == remote_calls
    df_remote.execute()
    assert vmin.get() == xmin
    assert vmax.get() == xmax
    assert df_remote.executor.remote_calls == remote_calls + 1

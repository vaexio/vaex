import vaex.misc.progressbar
import pytest
from unittest.mock import MagicMock
from common import *

def test_progress_bar():
    pb = vaex.misc.progressbar.ProgressBar(0, 100)
    pb.update(0)
    pb.update(50)
    assert "50.00%" in repr(pb)
    pb.finish()
    assert "elapsed time" in repr(pb)

def test_progress_bar_widget():
    pb = vaex.misc.progressbar.ProgressBarWidget(0, 100)
    pb.update(0)
    pb.update(50)
    assert "50.00%" in repr(pb)
    assert pb.bar.value == 50
    pb.finish()
    assert "elapsed time" in repr(pb)

@pytest.mark.parametrize("progress", ['vaex', 'widget'])
def test_progress(progress):
    df = vaex.from_arrays(x=vaex.vrange(0, 10000))
    df.sum('x', progress=progress)


def test_progress_cache():
    df = vaex.from_arrays(x=vaex.vrange(0, 10000))
    with vaex.cache.on():
        with vaex.progress.tree('vaex') as progressbar:
            df._set('x', progress=progressbar)
            assert progressbar.finished

        with vaex.progress.tree('rich') as progressbar:
            df._set('x', progress=progressbar)
            assert progressbar.children[0].finished
            assert progressbar.children[0].bar.status == 'from cache'


def test_progress_error():
    df = vaex.from_arrays(x=vaex.vrange(0, 10000))
    with vaex.progress.tree('rich') as progressbar:
        try:
            df._set('x', progress=progressbar, limit=1)
        except vaex.RowLimitException:
            pass
        assert progressbar.children[0].bar.status.startswith('Resulting hash_map_unique would')
        assert progressbar.children[0].finished
        assert progressbar.finished
        # assert progressbar.children[0].bar.status == 'from cache'


# progress only supported for local df's
def test_progress_calls(df, event_loop):
    with vaex.cache.off():
        x, y = df.sum([df.x, df.y], progress=True)
        counter = CallbackCounter(True)
        task = df.sum([df.x, df.y], delay=True, progress=counter)
        df.execute()
        x2, y2 = task.get()
        assert x == x2
        assert y == y2
        assert counter.counter > 0
        assert counter.last_args[0], 1.0


@pytest.mark.asyncio
async def test_progress_calls_async(df):
    with vaex.cache.off():
        x, y = df.sum([df.x, df.y], progress=True)
        counter = CallbackCounter(True)
        task = df.sum([df.x, df.y], delay=True, progress=counter)
        await df.executor.execute_async()
        x2, y2 = await task
        assert x == x2
        assert y == y2
        assert counter.counter > 0
        assert counter.last_args[0] == 1.0


def test_cancel(df):
    with vaex.cache.off():
        magic = MagicMock()
        df.executor.signal_cancel.connect(magic)
        def progress(f):
            return False
        with pytest.raises(vaex.execution.UserAbort):
            result = df.x.min(progress=progress)
            assert result is None
        magic.assert_called_once()


@pytest.mark.asyncio
async def test_cancel_async(df):
    with vaex.cache.off():
        magic = MagicMock()
        df.executor.signal_cancel.connect(magic)
        def progress(f):
            return False
        with pytest.raises(vaex.execution.UserAbort):
            value = df.x.min(progress=progress, delay=True)
            await df.execute_async()
            await value
        magic.assert_called_once()

# @pytest.mark.timeout(1)
def test_cancel_huge(client):
    with vaex.cache.off():
        df = client['huge']
        import threading
        main_thread = threading.current_thread()
        max_progress = 0
        def progress(f):
            nonlocal max_progress
            assert threading.current_thread() == main_thread
            max_progress = max(max_progress, f)
            return f > 0.01
        with pytest.raises(vaex.execution.UserAbort):
            assert df.x.min(progress=progress) is None
        # assert df.x.min() is not None
        assert max_progress < 0.1


@pytest.mark.asyncio
async def test_cancel_huge_async(client):
    with vaex.cache.off():
        df = client['huge']
        import threading
        main_thread = threading.current_thread()

        def progress(f):
            print("progress", f)
            assert threading.current_thread() == main_thread
            return f > 0.01
        with pytest.raises(vaex.execution.UserAbort):
            df.x.min(progress=progress, delay=True)
            await df.execute_async()
        # assert df.x.min() is not None

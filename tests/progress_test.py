import vaex.misc.progressbar
import pytest

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
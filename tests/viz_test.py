import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import pytest

import vaex


matplotlib.use('agg')


def test_histogram(df_trimmed):
    df = df_trimmed
    plt.figure()
    df.x.viz.histogram(show=False)


def test_heatmap(df_trimmed):
    df = df_trimmed
    plt.figure()
    df.x.viz.histogram(show=False)


# we should use this properly like mentioned in https://github.com/matplotlib/pytest-mpl
# for now it's only coverage for debugging failures (instead of failures on notebook runs)

@pytest.mark.mpl_image_compare(filename='test_histogram_with_what.png')
def test_histogram_with_what(df):
    fig = plt.figure()
    df.viz.histogram(df.x, what=np.clip(np.log(-vaex.stat.mean(df.z)), 0, 10), limits='99.7%');
    return fig

@pytest.mark.mpl_image_compare(filename='test_heatmap_with_what.png')
def test_heatmap_with_what(df):
    df.viz.heatmap(df.x, df.y, what=np.log(vaex.stat.count()+1), limits='99.7%',
        selection=[None, df.x < df.y, df.x < -10])

import matplotlib
import matplotlib.pyplot as plt

def test_histogram(df_trimmed):
    df = df_trimmed
    matplotlib.use('agg')
    plt.figure()
    df.x.viz.histogram(show=False)


def test_heatmap(df_trimmed):
    df = df_trimmed
    matplotlib.use('agg')
    plt.figure()
    df.x.viz.histogram(show=False)
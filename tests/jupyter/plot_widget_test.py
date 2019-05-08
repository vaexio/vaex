import vaex
import numpy as np


def test_plot_widget_bqplot():
    # basic coverage for now
    df = vaex.example()
    df.plot_widget(df.x, df.y)
    df.plot_widget(df.x.astype('float32'), df.y.astype('float32'))
    df.plot_widget(df.x.astype('float32'), df.y.astype('float32'), limits='minmax')
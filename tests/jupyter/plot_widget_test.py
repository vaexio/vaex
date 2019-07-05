import vaex
import numpy as np


def test_plot_widget_bqplot():
    # basic coverage for now
    df = vaex.example()
    df.plot_widget(df.x, df.y)
    df.plot_widget(df.x.astype('float32'), df.y.astype('float32'))
    df.plot_widget(df.x.astype('float32'), df.y.astype('float32'), limits='minmax')

def test_widget_hist():
    df = vaex.example()
    assert df.widget is df.widget
    hw = df.widget.histogram('x')


def test_widget_heatmap():
    df = vaex.example()
    hw = df.widget.heatmap('x', 'y')


def test_widget_pie():
    df = vaex.example()
    df['s'] = df.x < 0
    df.categorize('s', labels=['neg', 'pos'], check=False)
    hw = df.widget.pie('s')

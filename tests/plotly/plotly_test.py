import pytest
pytest.importorskip("plotly")

import vaex
import vaex.ml


df_iris = vaex.ml.datasets.load_iris()
df_example = vaex.example()


@pytest.mark.parametrize('x', ['sepal_width', ['sepal_width', 'sepal_length']])
def test_histogram(x):
    df_iris.plotly.histogram(x=x, shape=32, limits='minmax', color='red')


@pytest.mark.parametrize('color', ['red', df_iris.class_])
def test_scatter(color):
    df_iris.plotly.scatter(x=df_iris.sepal_width,
                           y=df_iris.sepal_length,
                           tooltip_title=df_iris.class_,
                           color=color,
                           colorbar=True,
                           title='title')


def test_scatter_multiple():
    df_iris.plotly.scatter(x=[df_iris.sepal_length, df_iris.sepal_width],
                           y=[df_iris.petal_length, df_iris.petal_width],
                           tooltip_title=df_iris.class_,
                           selection=['sepal_width > 0.5', 'petal_width < 0.5'],
                           color=['red', 'green'],
                           colorbar=True,
                           title='title')


@pytest.mark.parametrize('f', [None, 'log1p'])
@pytest.mark.parametrize('n', [None, 'normalize'])
@pytest.mark.parametrize('shape', [32, 128])
def test_heatmap(shape, f, n):
    df_example.plotly.heatmap(x=df_example.x,
                              y=df_example.y,
                              shape=shape,
                              colorbar_label='Colorbar Label',
                              colormap='jet',
                              f=f,
                              n=n,
                              selection='vx > 0')

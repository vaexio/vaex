import vaex
import vaex.ml
import tempfile
import vaex.ml.datasets
features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']


def test_pca(df_iris):
    ds = df_iris
    pca = vaex.ml.PCA(features=features, n_components=2)
    pca.fit(ds)
    ds1 = pca.transform(ds)

    path = tempfile.mktemp('.json')
    pipeline = vaex.ml.Pipeline([pca])
    pipeline.save(path)

    pipeline = vaex.ml.Pipeline()
    pipeline.load(path)
    ds2 = pipeline.transform(ds)
    assert ds1.virtual_columns['PCA_1'] == ds2.virtual_columns['PCA_1']

    path = tempfile.mktemp('.json')
    pipeline = vaex.ml.Pipeline([ds1.ml.state_transfer()])
    pipeline.save(path)

    pipeline = vaex.ml.Pipeline()
    pipeline.load(path)
    ds3 = pipeline.transform(ds)
    assert ds1.virtual_columns['PCA_1'] == ds3.virtual_columns['PCA_1']


def test_selections(df_iris):
    ds = df_iris
    ds.select('class_ == 1')
    count1 = ds.count(selection=True)

    path = tempfile.mktemp('.json')
    pipeline = vaex.ml.Pipeline([ds.ml.state_transfer()])
    pipeline.save(path)
    print(path)

    pipeline = vaex.ml.Pipeline()
    pipeline.load(path)
    ds2 = pipeline.transform(ds)
    assert ds2.count(selection=True) == count1


def test_state_transfer(df_iris):
    df_iris_copy = df_iris.copy()
    ds = df_iris
    ds['test'] = ds.petal_width * ds.petal_length
    test_values = ds.test.evaluate()
    state_transfer = ds.ml.state_transfer()

    # clean dataset
    ds = df_iris_copy
    ds = state_transfer.transform(ds)
    assert test_values.tolist() == ds.test.evaluate().tolist()

    ds1, ds2 = ds.split(0.5)
    state_transfer = ds1.ml.state_transfer()

    path = tempfile.mktemp('.json')
    pipeline = vaex.ml.Pipeline([state_transfer])
    pipeline.save(path)

import sys

import pytest

pytest.importorskip("vowpalwabbit")
from sklearn.metrics import accuracy_score
import vaex.ml.vowpalwabbit
import vaex.ml.datasets

params = {'oaa': '3', 'P': 1, 'enable_logging': True}


def test_vowpalwabbit(df_iris):
    ds = df_iris

    ds['class_'] = ds['class_'] + 1  # VW classification starts from 1
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z', 'w']

    params = {'oaa': '3', 'P': 1, 'link': 'logistic', 'enable_logging': True}
    model = vaex.ml.vowpalwabbit.VowpalWabbitModel(
        params=params,
        features=features,
        target='class_')
    model.fit(ds_train, passes=5)
    assert 0 < accuracy_score(ds_test.col.class_.values, model.predict(ds_test))

    ds_train = model.transform(ds_train)  # this will add the vw column
    state = ds_train.state_get()
    ds_test.state_set(state)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
def test_vowpalwabbit_serialize(tmpdir, df_iris):
    ds = df_iris
    ds['class_'] = ds['class_'] + 1  # VW classification starts from 1
    features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    model = vaex.ml.vowpalwabbit.VowpalWabbitModel(
        params=params,
        features=features,
        target='class_')
    model.fit(ds)
    pl = vaex.ml.Pipeline([model])
    pl.save(str(tmpdir.join('test.json')))
    pl.load(str(tmpdir.join('test.json')))

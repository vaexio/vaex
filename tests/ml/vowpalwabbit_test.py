import sys

import numpy as np
import pytest

pytest.importorskip("vowpalwabbit")

import vaex.ml.vowpalwabbit
import vaex.ml.datasets

params = {'oaa': '3', 'P': 1, 'enable_logging': True}


def accuracy_score(predictions, true_values):
    return (true_values == predictions).sum() / len(true_values)


def test_vowpalwabbit_examples(df_iris):
    ds = df_iris
    ds['class_'] = ds['class_'] + 1  # VW classification starts from 1
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1

    params = {'oaa': '3', 'P': 1, 'enable_logging': True}
    target = 'class_'
    model = vaex.ml.vowpalwabbit.VowpalWabbitModel(
        params=params,
        target=target)
    features = model._init_features(ds)
    assert model.target == target
    assert model.target not in features and len(features) == len(ds.get_column_names()) - 1

    features = ['x', 'y', 'z', 'w']
    model = vaex.ml.vowpalwabbit.VowpalWabbitModel(
        params=params,
        features=features,
        target=target)
    features = model._init_features(ds)
    assert model.target == target
    assert len(features) == 4
    from vowpalwabbit.DFtoVW import DFtoVW
    examples = DFtoVW.from_colnames(df=ds.head(1).to_pandas_df(), y=target, x=features).convert_df()
    examples[0] == '2 | x:5.9 y:3.0 z:1.5 w:4.2'


def test_vowpalwabbit(df_iris):
    ds = df_iris

    ds['class_'] = ds['class_'] + 1  # VW classification starts from 1
    ds['x'] = ds.sepal_length * 1
    ds['y'] = ds.sepal_width * 1
    ds['w'] = ds.petal_length * 1
    ds['z'] = ds.petal_width * 1
    ds_train, ds_test = ds.ml.train_test_split(test_size=0.2, verbose=False)
    features = ['x', 'y', 'z', 'w']

    params = {'oaa': '3', 'P': 1, 'enable_logging': True}
    model = vaex.ml.vowpalwabbit.VowpalWabbitModel(
        params=params,
        num_epochs=1,
        features=features,
        target='class_')
    model.fit(ds_train)
    score1 = accuracy_score(ds_test.col.class_.values, model.predict(ds_test))
    assert 0 < score1

    model.fit(ds_train)
    score2 = accuracy_score(ds_test.col.class_.values, model.predict(ds_test))

    model.fit(ds_train)
    score3 = accuracy_score(ds_test.col.class_.values, model.predict(ds_test))
    assert score1 < score2 < score3

    transformed = model.transform(ds_test)
    assert 'vowpalwabbit_prediction' in transformed


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

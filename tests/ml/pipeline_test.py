import tempfile

import vaex
import vaex.ml
import vaex.ml.datasets
from sklearn.linear_model import LogisticRegression
from vaex.dataframe import DataFrame
from vaex.ml.pipeline import Pipeline
from vaex.ml.sklearn import SKLearnPredictor

features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']


def test_pipeline_persistence():
    train, test = vaex.ml.datasets.load_iris().split_random(0.5)
    pca = vaex.ml.PCA(features=features, n_components=2)
    train = pca.fit_transform(train)
    pipeline = train.pipeline(train)
    transformed = pipeline.transform(test)
    assert 'PCA_0' in transformed
    assert 'PCA_1' in transformed

    path = tempfile.mktemp('.json')
    pipeline.save(path)
    pipeline = vaex.ml.pipeline.from_file(path)

    transformed = pipeline.transform(test)
    assert 'PCA_0' in transformed
    assert 'PCA_1' in transformed


def test_pipeline_selections():
    df = vaex.ml.datasets.load_iris()
    df['x'] = df.sepal_length + 1
    class1 = df[df.class_ == 1]
    assert len(class1) != len(df)

    pipeline = class1.pipeline()
    transformed = pipeline.transform(df)
    infered = pipeline.inference(df)
    assert len(transformed) == len(class1)  # apply state_set
    assert len(infered) == len(df)  # apply state_set without filtering
    assert 'x' in transformed
    assert 'x' in infered


def test_pipeline_predict():
    train, test = vaex.ml.datasets.load_iris().split_random(0.8)
    model = SKLearnPredictor(model=LogisticRegression(), features=features, target='class_')
    model.fit(train)
    train = model.transform(train)
    pipeline = Pipeline.from_dataset(train, predict_column='prediction')
    assert all(pipeline.predict(test) == model.predict(test))


def test_pipeline_predict():
    train, test = vaex.ml.datasets.load_iris().split_random(0.8)
    train = train[train.class_ != 2]
    model = SKLearnPredictor(model=LogisticRegression(), features=features, target='class_')
    model.fit(train)
    train = model.transform(train)
    pipeline = Pipeline.from_dataset(train, predict_column='prediction')
    transformed = pipeline.transform(test)
    assert 'prediction' in transformed
    assert len(transformed) < len(test)  # we applied the filter


def test_pipeline_inference():
    train, test = vaex.ml.datasets.load_iris().split_random(0.5)
    filtered = train[train.class_ != 2]

    model = SKLearnPredictor(model=LogisticRegression(), features=features, target='class_')
    model.fit(filtered)
    filtered = model.transform(filtered)
    pipeline = Pipeline.from_dataset(filtered, predict_column='prediction')
    assert len(pipeline.transform(test)) != len(pipeline.inference(test))
    assert len(pipeline.inference(test)) == len(test)


def test_pipeline_origin_columns():
    """
    What columns are needed to apply inference.
    If a column was not used at any point in the state, it is not an prerequisite in the new data.
    """
    train, test = vaex.ml.datasets.load_iris().split_random(0.8)
    train['x'] = train.sepal_length * 1
    pca = vaex.ml.PCA(features=['petal_length', 'sepal_length'], n_components=2)
    train = pca.fit_transform(train)
    pipeline = Pipeline.from_dataset(train)
    assert pipeline.origin_columns == ['sepal_length', 'petal_length']
    assert 'x' in pipeline.inference(test[['sepal_length', 'petal_length']])
    assert 'PCA_0' in pipeline.inference(test[['sepal_length', 'petal_length']])


def test_pipeline_fit():
    def fit(df):
        df['x'] = df.sepal_length * 1
        df['y'] = df.sepal_width * 1
        df['w'] = df.petal_length * 1
        df['z'] = df.petal_width * 1
        model = SKLearnPredictor(model=LogisticRegression(), features=['x', 'y', 'w', 'z'], target='class_')
        model.fit(df)
        df = model.transform(df)
        df['response'] = df['prediction'].map({0: 'class A', 1: 'class B', 2: 'class C'})
        return df

    train, test = vaex.ml.datasets.load_iris().split_random(0.8)

    train['a'] = train.sepal_length + 1
    pipeline = train.pipeline(fit=fit)
    assert pipeline.example

    assert 'a' in train
    assert 'prediction' not in train

    path = tempfile.mktemp('.json')
    pipeline.save(path)
    pipeline = vaex.ml.pipeline.from_file(path)

    'a' in pipeline.inference(test)
    'x' not in pipeline.inference(test)

    pipeline.fit(train)
    assert 'response' in pipeline.inference(test)

    assert len(pipeline.inference(test).prediction.unique()) == 3  # 3 classes

    # We will retrain the model on only two classes, as a result, only those two are predicted.
    filtered = train[train.class_ != 2]
    pipeline.fit(filtered)
    assert len(pipeline.inference(test).prediction.unique()) == 2

    pipeline.save(path)
    pipeline = Pipeline.from_file(path)

    transformed = pipeline.fit_transform(train)
    assert 'a' in transformed
    assert 'response' in transformed


def test_pipeline_infer():
    df = vaex.ml.datasets.load_iris()
    pdf = df.to_pandas_df()
    assert isinstance(Pipeline.infer(df), DataFrame)
    assert isinstance(Pipeline.infer(df.to_dict()), DataFrame)
    assert isinstance(Pipeline.infer(pdf), DataFrame)
    assert isinstance(Pipeline.infer(pdf.to_dict()), DataFrame)
    assert isinstance(Pipeline.infer(pdf.to_json()), DataFrame)
    assert isinstance(Pipeline.infer(pdf.iloc[0].to_dict()), DataFrame)  # single query {key:value}

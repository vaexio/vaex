import json
import pickle

import numpy as np
import pyarrow as pa
import vaex
from vaex.ml import Pipeline
from vaex.ml.datasets import load_titanic
from vaex.ml.lightgbm import LightGBMModel
from vaex.ml.transformations import Imputer


def test_pipeline_infer():
    df = vaex.ml.datasets.load_titanic().head(5)
    assert isinstance(Pipeline.infer(df), vaex.dataframe.DataFrame)  # Vaex
    assert isinstance(Pipeline.infer(df.to_pandas_df()), vaex.dataframe.DataFrame)  # pandas
    assert isinstance(Pipeline.infer(df.to_records()), vaex.dataframe.DataFrame)  # records: [{key:value,...}, ...]
    assert isinstance(Pipeline.infer(df.to_records(0)), vaex.dataframe.DataFrame)  # single records: {key:value,...}
    assert isinstance(Pipeline.infer(df.to_dict()), vaex.dataframe.DataFrame)  # vaex to_dict: {key:[value, value,...],}
    assert isinstance(Pipeline.infer(df.to_pandas_df().to_dict()),  # pandas to_dict: {key:{index: value,...],}
                      vaex.dataframe.DataFrame)
    assert isinstance(Pipeline.infer(df.to_pandas_df().to_json()),  # pandas json: '{key:{index:value,...},...}'
                      vaex.dataframe.DataFrame)

    # numpy with columns - maybe irrelevant
    assert isinstance(Pipeline.infer(df.to_pandas_df().to_numpy(), names=df.get_column_names()),
                      vaex.dataframe.DataFrame)  # numpy with columns
    assert isinstance(Pipeline.infer(df.to_pandas_df().to_numpy().T, names=df.get_column_names()),
                      vaex.dataframe.DataFrame)  # numpy transpose with columns
    # reading files
    from tempfile import TemporaryDirectory
    temp = TemporaryDirectory()
    df.export_hdf5(f"{temp.name}/titanic.hdf5")
    df.export_csv(f"{temp.name}/titanic.csv")
    assert isinstance(Pipeline.infer(f"{temp.name}/titanic.hdf5"),  # read file
                      vaex.dataframe.DataFrame)
    assert isinstance(Pipeline.infer(f"{temp.name}/titanic.csv", nrows=1),  # read file with params
                      vaex.dataframe.DataFrame)
    assert isinstance(Pipeline.infer(f"{temp.name}/"),  # read dir
                      vaex.dataframe.DataFrame)


def test_pipeline_inference():
    train, test = vaex.ml.datasets.load_iris().split_random(0.8)
    train['average_length'] = (train['sepal_length'] + train['petal_length']) / 2
    booster = LightGBMModel(features=['average_length', 'petal_width'], target='class_', num_boost_round=500,
                            params={'verbose': -1})
    booster.fit(train)
    train = booster.transform(train)

    pipeline = Pipeline.from_dataframe(train)

    assert pipeline.inference(test)['lightgbm_prediction'].mean() > 0
    assert pipeline.inference(test, columns='lightgbm_prediction')['lightgbm_prediction'].mean() > 0
    assert pipeline.inference(test[['sepal_length']], columns='lightgbm_prediction')['lightgbm_prediction'].mean() > 0

    assert isinstance(pipeline.inference(test.gl.to_records(0)), vaex.dataframe.DataFrame)


def test_pipeline_imputer(tmpdir):
    data = {
        'floats': [2.0, 3.1, 4.2, np.nan, np.nan],
        'bool': pa.array([True, True, True, False, None]),
        'int': pa.array([1, 2, 3, 4, None]),
        'str': ['a', 'b', 'b', 'c', None],
        'mean': [1, 2, 4, 5, np.nan],
        'int_common': pa.array([1, 1, 1, 2, None]),
        'int_new_value': pa.array([1, 1, 1, 2, np.nan]),
        'str_common': ['a', 'a', 'a', 'b', None],
        'mode': [1, 1, 2, 2, np.nan],
        'str_new_value': ['a', 'c', 'b', None, None],
        'int_new_value': pa.array([1, 2, 3, 4, None]),
        'str_value': ['a', 'a', 'a', 'a', None],
        'int_value': pa.array([1, 2, 3, 4, None]),
        'float_value': [1.0, 1.1, np.nan, np.nan, np.nan],
        'dict': [{}, {}, {}, {}, {}]
    }
    df = vaex.from_dict(data)
    assert df.gl.countna() == 17
    strategy = {
        int: 0,
        float: 0.0,
        str: 'NONE',
        'mean': Imputer.MEAN,
        'int_common': Imputer.COMMON,
        'int_new_value': Imputer.NEW_VALUE,
        'str_common': Imputer.COMMON,
        'mode': 'MODE',
        'str_new_value': Imputer.NEW_VALUE,
        'float_value': -2,
        'str_value': 'b',
    }
    imputer = Imputer(strategy=strategy)

    transformed = imputer.fit_transform(df)
    transformed['mean_plus1'] = transformed['mean'] + 1
    transformed['mean_plus2'] = transformed['mean_plus1'] + 1
    assert transformed.gl.countna() == 0

    pipeline = Pipeline.from_dataframe(transformed)

    assert set(pipeline.input_columns) == set(data.keys())
    assert pipeline.inference(df).gl.countna() == 0
    assert pipeline.example

    empty = vaex.from_arrays(x=[1])

    infered = pipeline.inference(empty)
    assert infered.gl.countna() == 0
    assert infered.shape[1] == transformed.shape[1] + 1 == df.shape[1] + 3  # { x, mean_plus1, mean_plus2 }

    # pickle
    new_pipeline = pickle.loads(pickle.dumps(pipeline))
    infered = new_pipeline.inference(df)
    assert infered.gl.countna() == 0

    # json
    new_pipeline = Pipeline.load_state(json.loads(json.dumps(pipeline.state_get())))
    assert new_pipeline.inference(df).gl.countna() == 0

    # save
    # from tempfile import TemporaryDirectory; # TODO remove
    # tmpdir = TemporaryDirectory().name  # TODO remove
    path = str(tmpdir.join('pipeline.pkl'))
    assert pipeline.from_file(pipeline.save(path)).transform(df).gl.countna() == 0


def test_pipeline_fit():
    df = vaex.from_arrays(a=[1, 2, 3], b=['a', 'a', 'b'])

    def fit(df):
        # TODO add model
        print('fit')
        df = df.copy()
        df['c'] = df['a'] + 1
        df['d'] = df['b'] + '!'
        return df

    trained = fit(df)
    pipeline = Pipeline.from_dataframe(df, fit=fit)
    pipeline.fit(df)
    transformed = pipeline.transform(df)
    assert 'c' in transformed
    assert 'd' in transformed

    pipeline = Pipeline.from_dataframe(df, fit=fit)
    pipeline.fit(df)
    transformed = pipeline.transform(df)
    assert 'c' in transformed
    assert 'd' in transformed
    pipeline.save('./test.pipeline')

    # try after restart
    df = vaex.from_arrays(a=[1, 1, 1], b=['a', 'b', 'a'])
    pipeline = Pipeline.from_file('./test.pipeline')
    pipeline.fit(df)


def test_advance_pipeline(tmpdir):
    def fit(train):
        import vaex
        from lightgbm import LGBMClassifier
        from vaex.ml.sklearn import Predictor
        from vaex.ml.transformations import LabelEncoder, MinMaxScaler, Imputer

        train['age_standard'] = (train.age - train.age.mean()) / train.age.std()  # by expression
        train['age_standard'] = train['age_standard'].expand().fillna(0)
        train['embarked'] = train.embarked.fillna("NA")
        train['age'] = train.age.fillna(train.age.mean())
        train['sex'] = train.sex.fillna('na')

        # # Some numeric fun
        train['age_standard'] = (train.age - train.age.mean()) / train.age.std()  # by expression

        train = MinMaxScaler(features=['fare']).fit_transform(train)  # by transformers
        # # apply
        train['sex_indicator'] = train['sex'].apply(lambda x: 0 if x == 'male' else 1)
        # # label encoder
        train = LabelEncoder(features=['embarked']).fit_transform(train)

        train = Imputer().fit_transform(train)
        assert train.gl.countna() == 0

        pca = vaex.ml.PCA(features=['sex_indicator', 'age_standard', 'minmax_scaled_fare', 'label_encoded_embarked'],
                          n_components=4)
        train = pca.fit_transform(train)

        features = train.get_column_names(regex='^PCA')

        model = Predictor(model=LGBMClassifier(), features=features, target='survived')

        model.fit(train)
        train = model.transform(train)
        train['results'] = train['prediction'].astype('int')
        return train

    train, test = load_titanic().split_random(0.7, random_state=42)
    # from tempfile import TemporaryDirectory; # TODO remove
    # tmpdir = TemporaryDirectory().name  # TODO remove
    pipeline = Pipeline.from_dataframe(train, fit=fit)
    pipeline.fit(train)
    path = str(tmpdir.join('pipeline.pkl'))
    new_pipeline = Pipeline.from_file(pipeline.save(path))
    assert 'results' in new_pipeline.inference(test)

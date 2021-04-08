import pytest
import vaex


@pytest.fixture(scope='session')
def df_iris_original():
    return vaex.ml.datasets.load_iris()


@pytest.fixture(scope='function')
def df_iris(df_iris_original, df_factory):
   return df_factory(**df_iris_original.to_dict())


@pytest.fixture(scope='session')
def df_iris_1e5_original():
    return vaex.ml.datasets.load_iris_1e5()


@pytest.fixture(scope='function')
def df_iris_1e5(df_iris_1e5_original, df_factory):
   return df_factory(**df_iris_1e5_original.to_dict())


@pytest.fixture(scope='session')
def df_titanic_original():
    return vaex.ml.datasets.load_titanic()


@pytest.fixture(scope='function')
def df_titanic(df_titanic_original, df_factory):
   return df_factory(**df_titanic_original.to_dict())



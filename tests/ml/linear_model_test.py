import pytest
pytest.importorskip("sklearn")
import sys
import numpy as np
import vaex.ml.linear_model
import vaex.ml.datasets

features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
def test_linear_model():
    ds = vaex.ml.datasets.load_iris()
    m1 = vaex.ml.linear_model.LinearRegression(features=['petal_width'], binned=False)
    m1.fit(ds, 'petal_length')
    # print(m.coef_, m.intercept_)
    m2 = vaex.ml.linear_model.LinearRegression(features=['petal_width'], binned=True)
    m2.fit(ds, 'petal_length')
    # print(m.coef_, m.intercept_)
    np.testing.assert_approx_equal(m1.intercept_, m2.intercept_, significant=2)
    np.testing.assert_approx_equal(np.array(m1.coef_), np.array(m2.coef_), significant=2)


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
@pytest.mark.skip(reason="This will fail: produces wrong answer")
def test_logit():
    ds = vaex.ml.datasets.load_iris()
    ds.categorize(ds.class_, labels='0 1 2 3'.split(), inplace=True)
    m1 = vaex.ml.linear_model.LogisticRegression(features=features, binned=False)
    m1.fit(ds, 'class_')
    class1 = m1.predict(ds)
    print(m1.coef_, m1.intercept_)
    m2 = vaex.ml.linear_model.LogisticRegression(features=features, binned=True, shape=32)
    m2.fit(ds, 'class_')
    class2 = m2.predict(ds)
    # print(m.coef_, m.intercept_)
    np.testing.assert_array_equal(class1, class2)

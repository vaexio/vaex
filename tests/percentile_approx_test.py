import vaex.ml
import numpy as np

def test_percentile_approx():
    df = vaex.ml.datasets.load_iris()
    p50 = df.percentile_approx('petal_width', 50)
    np.testing.assert_almost_equal(actual=p50, desired=1.301894, decimal=5)
    p25 = df.percentile_approx('petal_length', 25)
    np.testing.assert_almost_equal(actual=p25, desired=1.573027, decimal=5)
    p75 = df.percentile_approx('sepal_width', 75)
    np.testing.assert_almost_equal(actual=p75, desired=3.300586, decimal=5)
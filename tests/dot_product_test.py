import vaex
import numpy as np
from common import *

def test_dot_prodict(ds_local):
    df = ds_local
    result = df.func.dot_product([df.x, df.y], [2, 3])

    X = df['x', 'y'].values
    expected_result = np.dot(X, [2, 3])

    np.testing.assert_array_almost_equal(result.values, expected_result)
import numpy as np

import vaex


def test_dropinf():
    x = [1, 2, np.inf]
    y = [10, -np.inf, 2]
    z = [1, 2, 3]

    df = vaex.from_arrays(x=x, y=y, z=z)
    df_filter = df.dropinf()

    df_filter.shape == (1, 3)
    df_filter.values.tolist() == [[1.0, 10.0, 1.0]]

    df_filter = df.dropinf(column_names=['x'])
    df_filter.shape == (2, 3)
    df_filter.values.tolist() == [[1.0, 10.0, 1.0], [2.0, -np.inf, 2.0]]

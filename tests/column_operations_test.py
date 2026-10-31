import vaex

import numpy as np

import pytest


def test_add_str_and_numeric_types():
    x = np.arange(10)
    y = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    df = vaex.from_arrays(x=x, y=y)
    with pytest.raises(TypeError):
        df['z1'] = df.x + df.y
    with pytest.raises(TypeError):
        df['z2'] = df.y + df.z

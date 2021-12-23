from common import *
import pandas as pd
import pytest


@pytest.fixture()
def df():
    df = pd.DataFrame(
        {'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0], 'c': [True, True, False], 'd': ['a', 'b', 'c'], 'e': ['d', 'e', 'f']})
    df['e'] = df['e'].astype('category')
    df = vaex.from_pandas(df)
    return df


def test_get_column_names_dtypes(df):
    assert df.get_column_names(dtype=int) == ['a']
    assert df.get_column_names(dtype=float) == ['b']
    assert df.get_column_names(dtype=np.float64) == ['b']
    assert df.get_column_names(dtype=np.float32) == []
    assert df.get_column_names(dtype=[int, float]) == ['a', 'b']
    assert df.get_column_names(dtype=str) == ['d', 'e']
    assert df.get_column_names(dtype=bool) == ['c']


def test_getitem_by_dtypes(df):
    assert df[int].get_column_names() == ['a']
    assert df[float].get_column_names() == ['b']
    assert df[np.float64].get_column_names() == ['b']
    assert df[int, float].get_column_names() == ['a', 'b']
    assert df[str, bool].get_column_names() == ['c', 'd', 'e']

    with pytest.raises(KeyError) as e_info:
        df[np.float32]

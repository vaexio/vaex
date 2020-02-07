from common import *
import pytest


def test_copy(df):
    df = df
    df['v'] = df.x + 1
    df.add_variable('myvar', 2)

    dfc = df.copy()
    assert 'x' in dfc.get_column_names()
    assert 'v' in dfc.get_column_names()
    assert 'v' in dfc.virtual_columns
    assert 'myvar' in dfc.variables
    dfc.x.values

def test_non_existing_column(df_local):
    df = df_local
    with pytest.raises(NameError, match='.*Did you.*'):
        df.copy(column_names=['x', 'x_'])

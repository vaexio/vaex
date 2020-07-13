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


def test_copy_dependencies():
    df = vaex.from_scalars(x=1)
    df['y'] = df.x + 1
    df2 = df.copy(['y'])
    assert df2.get_column_names(hidden=True) == ['y', '__x']
    # why.. ?
    #assert df2[['y']].get_column_names(hidden=True) == ['y', '__y']


def test_copy_dependencies_invalid_identifier():
    df = vaex.from_dict({'#': [1]})
    # add a virtual column depending on a real column with invalid identifier
    # this means the expression is non-trivial df['#']
    df['y'] = df['#'] + 1
    df2 = df.copy(['y'])
    assert df2.get_column_names(hidden=True) == ['y', "__#"]

    # add a virtual column with an invalid identifier
    df['$'] = df['#'] + df['y']
    df['z'] = df['y'] + df['$']
    df2 = df.copy(['z'])
    assert set(df2.get_column_names(hidden=True)) == {'z', "__#", "__$", '__y'}

    # copy also takes expressions
    df['@'] = df['y'] + df['$']
    df2 = df.copy(["df['@'] * 2"])
    assert set(df2.get_column_names(hidden=True)) == {"df['@'] * 2", '__@', "__#", "__$", '__y'}

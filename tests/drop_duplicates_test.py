import pytest
import vaex
import numpy as np

@pytest.fixture()
def df():
    return vaex.from_arrays(a=[1, 2, 1, 3, 2], b=[1, 3, 1, 1, 2], c=[2, 2, 2, 4, 5])


def test_all_columns(df):
    all_drop = df.drop_duplicates(df.get_column_names())
    default_drop = df.drop_duplicates()

    assert (all_drop.values == default_drop.values).all()
    assert len(all_drop) == 4
    assert '__hidden_count' not in default_drop.get_column_names(hidden=True)


def test_drop_single_column(df):
    dropped = df.drop_duplicates(['a'])

    assert len(dropped) == 3
    assert list(sorted(dropped['a'].values)) == [1, 2, 3]


def test_drop_single_with_string(df):
    string_dropped = df.drop_duplicates('a')
    array_dropped = df.drop_duplicates(['a'])

    (string_dropped.values == array_dropped.values).all()


def test_drop_multiple_columns(df):
    dropped = df.drop_duplicates(['a', df.b])

    assert len(dropped) == 4


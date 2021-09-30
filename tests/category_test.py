from common import *
import collections
import numpy as np
import pyarrow as pa
import vaex
import pytest


@pytest.mark.parametrize("auto_encode", [False, True])
def test_cat_string(auto_encode):
    ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'])
    ds = ds0.ordinal_encode('colors')#, ['red', 'green'], inplace=True)
    assert ds.colors.dtype.internal.name == 'int8'
    ds = ds._future() if auto_encode else ds
    assert ds.is_category('colors')
    if auto_encode:
        assert ds.data_type('colors') == str
    else:
        assert ds.data_type('colors') == int
    assert ds.limits('colors', shape=128) == ([-0.5, 2.5], 3)

    ds = ds0.ordinal_encode('colors', values=['red', 'green'])
    assert ds.is_category('colors')
    assert ds.limits('colors', shape=128) == ([-0.5, 1.5], 2)
    assert ds.data.colors.tolist() == [0, 1, None, 1]

    assert ds.copy().is_category(ds.colors)

    # with pytest.raises(ValueError):
    # 	assert ds.is_category('colors', values=['red', 'orange'])

def test_count_cat():
    ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'], counts=[1, 2, 3, 4])
    ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'], names=['apple', 'apple', 'berry', 'apple'])
    ds = ds0.ordinal_encode(ds0.colors)
    ds = ds0.ordinal_encode(ds0.names)

    ds = ds0.ordinal_encode('colors', ['red', 'green', 'blue'])
    assert ds.count(binby=ds.colors).tolist() == [1, 2, 1]
    ds = ds0.ordinal_encode('colors', ['red', 'blue', 'green', ], inplace=True)
    assert ds.count(binby=ds.colors).tolist() == [1, 1, 2]


def test_categorize():
    ds0 = vaex.from_arrays(c=[0, 1, 1, 3])
    ds0.categorize('c', labels=['a', 'b', 'c', 'd'], inplace=True)
    assert ds0.is_category(ds0.c)
    assert ds0.category_labels(ds0.c) == ['a', 'b', 'c', 'd']
    assert ds0.category_count(ds0.c) == 4


def test_cat_missing_values():
    colors = ['red', 'green', 'blue', 'green', 'MISSING']
    mask   = [False, False,   False,   False,  True]
    colors = np.ma.array(colors, mask=mask)
    ds0 = vaex.from_arrays(colors=colors)
    ds = ds0.ordinal_encode('colors', ['red', 'green', 'blue'])
    assert ds.count(binby=ds.colors, edges=True).tolist() == [1, 0, 1, 2, 1, 0]

    # if we want missing values and non-categorized values to be reported seperately
    # the following is expected
    # ds = ds0.ordinal_encode('colors', ['red', 'green'])
    # assert ds.count(binby=ds.colors, edges=True).tolist() == [1, 0, 1, 2, 0, 1]


def test_categorize_integers():
    df = vaex.from_arrays(x=range(5, 15))
    df.categorize('x', min_value=5, labels=range(5, 15), inplace=True)
    assert df.count(binby='x').tolist() == [1] * 10
    assert df.binby('x', 'count').data.tolist() == [1] * 10

    df = vaex.from_arrays(x=range(5, 15))
    df.categorize('x', inplace=True)  # same, but calculated from data
    assert df.count(binby='x').tolist() == [1] * 10
    assert df.binby('x', 'count').data.tolist() == [1] * 10


@pytest.mark.parametrize("auto_encode", [False, True])
def test_cat_compare(df_factory, auto_encode):
    df = df_factory(x=np.array([0, 1, 2, 0], dtype='uint8'))
    df = df.categorize('x', labels=['a', 'b', 'c'])
    df = df._future() if auto_encode else df
    if auto_encode:
        assert df['x'].tolist() == ['a', 'b', 'c', 'a']
        assert str(df.x == 'a') == '(index_values(x) == 0)'
        assert df[df.x == 'a'].x.tolist() == ['a', 'a']
        with pytest.raises(ValueError):
            assert str(df.x == 'x') == '(x == 0)'
    else:
        # assert df['x'].tolist() == [0, 1, 2, 0]
        # # assert str(df.x == 'a') == '(x == 0)'
        assert df['x'].tolist() == [0, 1, 2, 0]
        assert str(df.x == 0) == '(x == 0)'
        assert df[df.x == 0].x.tolist() == [0, 0]


def test_arrow_dict_encoded(df_factory_arrow):
    indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
    dictionary = pa.array(['aap', 'noot', 'mies'])
    c = pa.DictionaryArray.from_arrays(indices, dictionary)
    df = df_factory_arrow(c=c)
    assert df.category_labels('c') == ['aap', 'noot', 'mies']
    assert df.category_count('c') == 3
    assert df.category_offset('c') == 0


def test_index_values(df_factory_arrow):
    indices = pa.array([0, 0, 1, 2])
    dictionary = pa.array(['aap', 'noot', 'mies'])
    c = pa.DictionaryArray.from_arrays(indices, dictionary)
    c = pa.chunked_array([c[i:i+1] for i in range(len(c))])
    df = df_factory_arrow(c=c)
    df = df._future()
    # assert df.c.index_values().tolist() == [0, 0, 1, 2]
    with small_buffer(df, 2):
        assert df[df.c == 'aap'].c.index_values().tolist() == [0, 0]


def test_ordinal_encode_optimize():
    x = np.random.choice(2, 10, replace=True)
    df = vaex.from_arrays(x=x)
    with pytest.warns(UserWarning, match='.*categorize.*'):
        df.ordinal_encode(df.x)

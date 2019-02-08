from common import *
import collections
import numpy as np
import vaex
import pytest

def test_cat_string():
    ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'])
    ds = ds0.ordinal_encode('colors')#, ['red', 'green'], inplace=True)
    assert ds.iscategory('colors')
    assert ds.limits('colors', shape=128) == ([-0.5, 2.5], 3)

    ds = ds0.ordinal_encode('colors', values=['red', 'green'])
    assert ds.iscategory('colors')
    assert ds.limits('colors', shape=128) == ([-0.5, 1.5], 2)
    assert ds.data.colors.tolist() == [0, 1, None, 1]

    assert ds.copy().iscategory(ds.colors)

    # with pytest.raises(ValueError):
    # 	assert ds.iscategory('colors', values=['red', 'orange'])

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
    ds0.categorize('c', ['a', 'b', 'c', 'd'])
    assert ds0.iscategory(ds0.c)
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

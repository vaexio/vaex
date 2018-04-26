from common import *
import collections
import numpy as np
import vaex
import pytest

def test_cat_string():
	ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'])
	ds = ds0.label_encode('colors')#, ['red', 'green'], inplace=True)
	assert ds.iscategory('colors')
	assert ds.limits('colors', shape=128) == ([-0.5, 2.5], 3)

	ds = ds0.label_encode('colors', values=['red', 'green'])
	assert ds.iscategory('colors')
	assert ds.limits('colors', shape=128) == ([-0.5, 1.5], 2)
	assert ds.data.colors.tolist() == [0, 1, None, 1]

	assert ds.copy().iscategory(ds.colors)

	# with pytest.raises(ValueError):
	# 	assert ds.iscategory('colors', values=['red', 'orange'])

def test_count_cat():
	ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'], counts=[1, 2, 3, 4])
	ds0 = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'], names=['apple', 'apple', 'berry', 'apple'])
	ds = ds0.label_encode(ds0.colors)
	ds = ds0.label_encode(ds0.names)	

	ds = ds0.label_encode('colors', ['red', 'green', 'blue'])
	assert ds.count(binby=ds.colors).tolist() == [1, 2, 1]
	ds = ds0.label_encode('colors', ['red', 'blue', 'green', ], inplace=True)
	assert ds.count(binby=ds.colors).tolist() == [1, 1, 2]



# def test_plot_cat():
# 	ds = vaex.from_arrays(colors=['red', 'green', 'blue', 'green'], counts=[4, ])
# 	ds.categorize('colors', inplace=True)#, ['red', 'green'], inplace=True)

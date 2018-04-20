import vaex
from common import *
import numpy as np

def test_basics(ds):
	x = ds['x']
	y = ds['y']
	z = x * y
	assert z.derivative(y).expression == 'x'

	z = x ** 2
	assert z.derivative(x).expression == '(2 * x)'

	z = np.sqrt(x)
	assert z.derivative(x).expression == '(0.5 * (x ** -0.5))'

	z = x / y
	assert z.derivative(y).expression == '(-x / (y ** 2))'

	z = x - y
	assert z.derivative(x).expression == '1'
	assert z.derivative(y).expression == '-1'

	z = np.log10(x)
	assert z.derivative(x).expression == '(1 / (x * log(10)))'	
	assert z.derivative(y).expression == '0'

	z = np.log10(x**2)
	assert z.derivative(x).expression == '((1 / ((x ** 2) * log(10))) * (2 * x))'	
	assert z.derivative(y).expression == '0'

	z = np.arctan2(y, x)
	assert z.derivative(x).expression == '(-y / ((x ** 2) + (y ** 2)))'
	assert z.derivative(y).expression == '(x / ((x ** 2) + (y ** 2)))'

def test_propagate_uncertainty():
	pass
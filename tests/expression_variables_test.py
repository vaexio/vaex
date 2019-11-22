import vaex
import numpy as np

def test_expression_expand():
	ds = vaex.from_scalars(x=1, y=2)
	ds['r'] = ds.x * ds.y
	assert ds.r.expression == 'r'
	assert ds.r.variables() == {'x', 'y'}
	ds['s'] = ds.r + ds.x
	assert ds.s.variables() == {'r', 'x', 'y'}
	assert ds.s.variables(ourself=True) == {'s', 'r', 'x', 'y'}
	assert ds.s.variables(include_virtual=False) == {'x', 'y'}
	assert ds.s.variables(ourself=True, include_virtual=False) == {'s', 'x', 'y'}
	ds['t'] = ds.s + ds.y
	assert ds.t.variables() == {'s', 'r', 'x', 'y'}
	ds['u'] = np.arctan(ds.t)
	assert ds.u.variables() == {'t', 's', 'r', 'x', 'y'}

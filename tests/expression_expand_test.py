import vaex
import numpy as np

def test_expression_expand():
	ds = vaex.from_scalars(x=1, y=2)
	ds['r'] = ds.x * ds.y
	assert ds.r.expression == 'r'
	assert ds.r.expand().expression == '(x * y)'
	ds['s'] = ds.r + ds.x
	assert ds.s.expand().expression == '((x * y) + x)'
	ds['t'] = ds.s + ds.y
	assert ds.t.expand(stop=['r']).expression == '((r + x) + y)'
	ds['u'] = np.arctan2(ds.s, ds.y)
	assert ds.u.expand(stop=['r']).expression == 'arctan2((r + x), y)'


def test_invert():
	df = vaex.from_scalars(x=1, y=2)
	df['r'] = ~(df.x > df.y)
	df.r.expand().expression == '~(x > y)'

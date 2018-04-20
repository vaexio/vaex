import vaex

def test_propagate_uncertainty():
	ds = vaex.from_scalars(x=1, y=2, e_x=2, e_y=4)
	ds['r'] = ds.x +  ds.y
	ds.propagate_uncertainties([ds.r])
	print(ds.r_uncertainty.expression)
	assert ds.r_uncertainty.expand().expression == 'sqrt(((e_x ** 2) + (e_y ** 2)))'

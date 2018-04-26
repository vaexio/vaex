import vaex

def test_propagate_uncertainty():
	ds = vaex.from_scalars(x=1, y=2, e_x=2, e_y=4)
	ds['r'] = ds.x +  ds.y
	ds.propagate_uncertainties([ds.r])
	print(ds.r_uncertainty.expression)
	assert ds.r_uncertainty.expand().expression == 'sqrt(((e_x ** 2) + (e_y ** 2)))'


def test_matrix():
	ds = vaex.from_scalars(x=1, y=0, z=0, x_e=0.1, y_e=0.2, z_e=0.3)
	matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
	ds.add_virtual_columns_matrix3d(ds.x, ds.y, ds.z, 'xn', 'yn', 'zy', matrix)
	ds.propagate_uncertainties([ds.xn])
	assert ds.xn.values[0] == ds.x.values[0]
	assert ds.xn_uncertainty.values[0] == ds.x_e.values[0]

	ds = vaex.from_scalars(x=1, y=0, z=0, x_e=0.1, y_e=0.2, z_e=0.3)
	matrix = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
	ds.add_virtual_columns_matrix3d(ds.x, ds.y, ds.z, 'xn', 'yn', 'zy', matrix)
	ds.propagate_uncertainties([ds.xn, ds.yn])
	assert ds.xn.values[0] == ds.y.values[0]
	assert ds.xn_uncertainty.values[0] == ds.y_e.values[0]

	assert ds.yn.values[0] == ds.x.values[0]
	assert ds.yn_uncertainty.values[0] == ds.x_e.values[0]

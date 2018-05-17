from common import *

def test_export(ds_local, tmpdir):
	ds = ds_local
	path = str(tmpdir.join('test.hdf5'))
	ds.export_hdf5(path)
	ds = ds.sample(5)
	path = str(tmpdir.join('sample.hdf5'))
	ds.export_hdf5(path)
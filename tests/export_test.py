from common import *
import os

def test_export(ds_local, tmpdir):
	ds = ds_local
	# TODO: we eventually want to support dtype=object, but not for hdf5
	ds = ds.drop(ds.obj)
	path = str(tmpdir.join('test.hdf5'))
	ds.export_hdf5(path)
	ds = ds.sample(5)
	path = str(tmpdir.join('sample.hdf5'))
	ds.export_hdf5(path)
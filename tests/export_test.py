from common import *
import os
import tempfile


@pytest.mark.parametrize("filename", ["test.hdf5", "test.arrow"])
def test_export_empty_string(tmpdir, filename):
	path = str(tmpdir.join('test.hdf5'))
	s = np.array(["", ""])
	df = vaex.from_arrays(s=s)
	df.export(path)
	vaex.open(path)


def test_export(ds_local, tmpdir):
	ds = ds_local
	# TODO: we eventually want to support dtype=object, but not for hdf5
	ds = ds.drop(ds.obj)
	path = str(tmpdir.join('test.hdf5'))
	ds.export_hdf5(path)
	ds = ds.sample(5)
	path = str(tmpdir.join('sample.hdf5'))
	ds.export_hdf5(path)

	ds = ds.drop(ds.timedelta)

	path = str(tmpdir.join('sample.parquet'))
	ds.export(path)
	df = vaex.open(path)

def test_export_open_hdf5(ds_local):
	ds = ds_local
	ds = ds.drop(ds.obj)
	filename = tempfile.mktemp(suffix='.hdf5')
	ds.export(filename)
	ds_opened = vaex.open(filename)
	assert list(ds) == list(ds_opened)


def test_export_open_hdf5(ds_local):
	ds = ds_local
	ds = ds.drop(ds.obj)
	ds = ds.drop(ds.timedelta)
	filename = tempfile.mktemp(suffix='.arrow')
	ds.export(filename)
	ds_opened = vaex.open(filename)
	assert list(ds) == list(ds_opened)


def test_export_string_mask(tmpdir):
	df = vaex.from_arrays(s=vaex.string_column(['aap', None, 'mies']))

	path = str(tmpdir.join('test.hdf5'))
	df.export(path)
	df_hdf5 = vaex.open(path)
	assert df.s.tolist() == df_hdf5.s.tolist()

	path = str(tmpdir.join('test.arrow'))
	df.export(path)
	df_arrow = vaex.open(path)
	assert df.s.tolist() == df_arrow.s.tolist()


# N = 2**32+2
# @pytest.mark.skipif(not os.environ.get('VAEX_EXPORT_BIG', False),
#                     reason="only runs when the env var VAEX_EXPORT_BIG is defined")
# def test_export_big(tmpdir):
# 	path = str(tmpdir.join('test.hdf5'))
# 	s = np.zeros(N, dtype='U1')
# 	s[:] = 'x'
# 	s[-1] = 'y'
# 	ds = vaex.from_arrays(s=s)
# 	ds.export_hdf5(path)
# 	df = ds.open(path)
# 	assert df[0:2].s.tolist() == ['x', 'x']
# 	assert df[-3:-1].s.tolist() == ['y', 'y']

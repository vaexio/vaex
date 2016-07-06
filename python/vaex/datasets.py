__author__ = 'maartenbreddels'
import sys
import vaex.utils
import vaex as vx
import os
import pandas as pd
data_dir = "/tmp/vaex/data"
if not os.path.exists(data_dir):
	os.makedirs(data_dir)


def _url_to_filename(url, replace_ext=None):
	filename = os.path.join(data_dir, url.split("/")[-1])
	if replace_ext:
		dot_index = filename.rfind(".")
		filename = filename[:dot_index] + "." + replace_ext
	return filename


class NYCTaxi(object):
	def __init__(self, name, url_list):
		self.name = name
		self.url_list = url_list

	@property
	def filename_single(self):
		return os.path.join(data_dir, self.name+".hdf5")

	@property
	def filenames(self):
		return map(_url_to_filename, self.url_list)

	@property
	def filenames_vaex(self):
		return map(lambda x: _url_to_filename(x, ".hdf5"), self.url_list)

	def get_single(self):
		ds = self.get()
		if len(self.filenames) > 1:
			if not os.path.exists(self.filename_single):
				ds.export_hdf5(self.filename_single)
			ds = vx.open(self.filename_single)
		return ds

	def get(self):
		self.download()
		self.fix()
		self.convert()
		return self.open()

	def download(self, force=False):
		for i, url in enumerate(self.url_list):
			if not os.path.exists(self.filenames[i]) or force:
				print("Downloading %s (%d out of %d)" % (url, i+1, len(self.url_list)))
				code = os.system("wget -c -P %s %s" % (data_dir, url))
				if code != 0:
					raise RuntimeError("wget finished with an error")

	def fix(self):
		pass

	def convert(self, force=False):
		skips = ["store_and_fwd_flag"]
		for i, (input, output) in enumerate(zip(self.filenames, self.filenames_vaex)):
			date_names = ["tpep_pickup_datetime","tpep_dropoff_datetime"]
			if not os.path.exists(output) or force:
				print("Converting %s to %s (%d out of %d)" % (input, output, i+1, len(self.filenames)))
				df = pd.read_csv(input, parse_dates=date_names)

				for skip in skips:
					if skip in df:
						del df["store_and_fwd_flag"]
				ds = vx.from_pandas(df)
				ds.add_virtual_column("pickup_hour", "hourofday(tpep_pickup_datetime)")
				ds.add_virtual_column("dropoff_hour", "hourofday(tpep_dropoff_datetime)")
				ds.add_virtual_column("pickup_dayofweek", "dayofweek(tpep_pickup_datetime)")
				ds.add_virtual_column("dropoff_dayofweek", "dayofweek(tpep_dropoff_datetime)")
				ds.export_hdf5(output, virtual=True)

	def open(self):
		return vx.open_many(self.filenames_vaex) if len(self.filenames_vaex) != 1 else vx.open(self.filenames_vaex[0])

urllist = """https://storage.googleapis.com/tlc-trip-data/2013/green_tripdata_2013-08.csv
https://storage.googleapis.com/tlc-trip-data/2013/green_tripdata_2013-09.csv
https://storage.googleapis.com/tlc-trip-data/2013/green_tripdata_2013-10.csv
https://storage.googleapis.com/tlc-trip-data/2013/green_tripdata_2013-11.csv
https://storage.googleapis.com/tlc-trip-data/2013/green_tripdata_2013-12.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-01.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-02.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-03.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-04.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-05.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-06.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-07.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-08.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-09.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-10.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-11.csv
https://storage.googleapis.com/tlc-trip-data/2014/green_tripdata_2014-12.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-01.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-02.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-03.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-04.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-05.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-06.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-07.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-08.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-09.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-10.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-11.csv
https://storage.googleapis.com/tlc-trip-data/2015/green_tripdata_2015-12.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-01.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-02.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-03.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-04.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-05.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-06.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-07.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-08.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-09.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-10.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-11.csv
https://storage.googleapis.com/tlc-trip-data/2009/yellow_tripdata_2009-12.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-01.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-02.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-03.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-04.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-05.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-06.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-07.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-08.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-09.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-10.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-11.csv
https://storage.googleapis.com/tlc-trip-data/2010/yellow_tripdata_2010-12.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-01.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-02.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-03.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-04.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-05.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-06.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-07.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-08.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-09.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-10.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-11.csv
https://storage.googleapis.com/tlc-trip-data/2011/yellow_tripdata_2011-12.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-01.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-02.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-03.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-04.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-05.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-06.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-07.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-08.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-09.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-10.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-11.csv
https://storage.googleapis.com/tlc-trip-data/2012/yellow_tripdata_2012-12.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-01.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-02.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-03.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-04.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-05.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-06.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-07.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-08.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-09.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-10.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-11.csv
https://storage.googleapis.com/tlc-trip-data/2013/yellow_tripdata_2013-12.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-01.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-02.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-03.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-04.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-05.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-06.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-07.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-08.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-09.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-10.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-11.csv
https://storage.googleapis.com/tlc-trip-data/2014/yellow_tripdata_2014-12.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-01.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-02.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-03.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-04.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-05.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-06.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-07.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-08.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-09.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-10.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-11.csv
https://storage.googleapis.com/tlc-trip-data/2015/yellow_tripdata_2015-12.csv""".split("\n")

nyctaxi_2015_jan = NYCTaxi("nyc_taxi2015jan", [urllist[-12]])
nyctaxi_2015 = NYCTaxi("nyc_taxi2015", urllist[-12:])

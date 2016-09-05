__author__ = 'breddels'
__author__ = 'breddels'
# due to 32 bit limitations in numpy, we cannot use astropy's fits module for writing colfits
import sys
import math

import pandas as pd
import vaex.dataset
import psutil

def meminfo():
	vmem = psutil.virtual_memory()
	print(("total mem", vmem.total/1024.**3, "avail", vmem.available/1024.**3))

def test_pandas(dataset):
	#f = open(path, "wb")
	meminfo()
	index = dataset.columns["random_index"]
	x = pd.Series(dataset.columns["x"], index=index)
	y = pd.Series(dataset.columns["y"], index=index)
	z = pd.Series(dataset.columns["z"], index=index)
	f = pd.DataFrame({"x": x, "y":y, "z":z})
	print((f.x.mean()))
	print((f.y.mean()))
	print((f.z.mean()))
	meminfo()

	#y = pd.Series(dataset.columns["x"])




if __name__ == "__main__":
	input = sys.argv[1]
	#output = sys.argv[2]
	dataset_in = vaex.dataset.load_file(input)
	test_pandas(dataset_in)

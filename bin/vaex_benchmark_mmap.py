from __future__ import print_function
__author__ = 'breddels'
from ctypes import *
import h5py
import sys
import numpy as np
import mmap
import vaex
import vaex.vaexfast
import timeit
import threading


filename = sys.argv[1]
h5file = h5py.File(filename, "r")

column = h5file[sys.argv[2]]
length = len(column)
assert column.dtype == np.float64
offset = column.id.get_offset()

h5file.close()

file = open(filename)
fileno = file.fileno()
dtype = np.float64
mapping = mmap.mmap(fileno, 0, prot=mmap.PROT_READ)
mmapped_array = np.frombuffer(mapping, dtype=dtype, count=length, offset=offset)

N = 3
byte_size = length * 8



thread_local = threading.local()
import ctypes.util
libc = cdll.LoadLibrary(ctypes.util.find_library('c'))
#fopen = libc.fopen
#fread = libc.fread
#cread = libc.read
#fread.argtypes = [c_void_p, c_size_t, c_size_t, c_void_p]
#fdopen = libc.fdopen
#fh = fdopen(fileno, "r")

def sum_read_part(i1, i2):
	if not hasattr(thread_local, "buffer"):
		#thread_local.buffer = np.zeros((i2-i1), dtype=np.float64)
		thread_local.c_buffer = ctypes.create_string_buffer((i2-i1)*8)
		thread_local.buffer = np.frombuffer(thread_local.c_buffer)
		# opening the file for each thread avoids having mutexes slow us down in the c code
		#thread_local.file = open(filename)
		#thread_local.fileno = thread_local.file.fileno()
		thread_local.file = open(filename, "r", 1)
		#thread_local.fileno = thread_local.file.fileno()

		#print "creating buffer"
	c_buffer = thread_local.c_buffer
	buffer = thread_local.buffer
	buffer = buffer[:i2-i1] # clip it if needed
	thread_local.file.seek(offset+i1)
	buffer = np.fromfile(thread_local.file, count=i2-i1+1)
	#thread_local.read()
	#fread(c_buffer, 8, (i2-i1), fh)
	#libc.read(thread_local.fileno, c_buffer, (i2-i1)*8)
	#data = file.read((i2-i1)*8)
	#buffer = np.fromstring(data, dtype=np.float64)
	return np.sum(buffer)
	#return vaex.vaexfast.sum(mmapped_array[i1:i2])

import concurrent.futures

def sum_read():
	total = sum([future.result() for future in vaex.utils.submit_subdivide(9, sum_read_part, length, int(2e5))])
	return total

#for i in range(3):
#	print sum_read()
print("benchmarking read mmap", sum_read())
expr = "sum_read()"
print(sum_read())
times = timeit.repeat(expr, setup="from __main__ import sum_read", repeat=5, number=N)
print("minimum time", min(times)/N)
bandwidth = [byte_size/1024.**3/(time/N) for time in times]
print("%f GiB/s" % max(bandwidth))







def sum_mmap_part(i1, i2):
	return np.sum(mmapped_array[i1:i2])
	#return vaex.vaexfast.sum(mmapped_array[i1:i2])

import concurrent.futures

def sum_mmap():
	total = sum([future.result() for future in vaex.utils.submit_subdivide(8, sum_mmap_part, length, int(1e6))])
	return total

print("benchmarking sum mmap", sum_mmap(), sum_mmap(), sum_mmap())
expr = "sum_mmap()"
print(sum_mmap())
times = timeit.repeat(expr, setup="from __main__ import sum_mmap", repeat=5, number=N)
print("minimum time", min(times)/N)
bandwidth = [byte_size/1024.**3/(time/N) for time in times]
print("%f GiB/s" % max(bandwidth))




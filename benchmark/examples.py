import numpy as np
import vaex as vx
import numexpr as ne
import vaex.multithreading as mt
import timeit
import math
import vaex.execution
import threading
lock = threading.Lock()
import sys

pool = mt.pool
ds = vx.open("data/Aq-A-2-999-shuffled-10percent.hdf5") if len(sys.argv) == 1 else sys.argv[1]
x = ds("x")
xlim = x.minmax()
data = ds.columns["x"]
print len(data), len(data)/4, len(data)%4, math.ceil(float(len(data))/pool.nthreads)
splits = 10
buf_size = int(1e8)
buf = np.zeros((pool.nthreads, len(data)/pool.nthreads+10), dtype=np.float64)
print buf.shape
import concurrent.futures
import theano.tensor as T
from theano import function
x = T.dvector('x')
z = eval("x**2")
func = function([x], z)
def case_a():
	#executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
	def f(thread_index, i1, i2):
		#size = i2-i1
		#print thread_index, size, i1, i2

		results = []
		#funct = func.__copy__()

		start = i1
		end = min(i1+buf_size, i2)
		done = False
		while not done:
			#print "from", start, "to", end
			size = end-start
			buf_thread = buf[thread_index][0:size]
			if 1:
				with lock:
					ne.evaluate("x**2", local_dict=dict(x=data[start:end]), out=buf_thread)
			else:
				#with lock:
				funct = function([x], z, outpu)
				buf_thread = funct(data[start:end])

			results.append(xlim.map(start, end, buf_thread))
			#print "done"
			done = end == i2
			start, end = end, min(end+buf_size, i2)
		#return xlim.map(i1, i2, buf_thread)
		#return xlim.map(i1, i2, data[i1:i2])
		return xlim.reduce(results)
	results = pool.run_blocks(f, len(data))
	#print results
	#results = mt.run_blocks(xlim.map, data)
	result = xlim.reduce(results)
	#xlim = result
	#print xlim
	return result

print case_a()
#dsa
def case_b():
	return vx.execution.main_manager.find_min_max(ds, ["x**2"])
	#vx.execution.main_manager.execute()

print case_b()
N = 20
print "case_a", timeit.timeit("case_a()", number=N, setup="from __main__ import case_a")/N
print "case_b", timeit.timeit("case_b()", number=N, setup="from __main__ import case_b")/N

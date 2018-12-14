from __future__ import print_function
import vaex as vx
import sys
import timeit
import numpy as np


progressbar = False
fn = sys.argv[1]
print(("opening", fn))
dataset = vx.open(fn)
#dataset = vx.open_many(fn)

expressions = sys.argv[2:]
print("subspace", expressions)
subspace = dataset(*expressions)
byte_size = len(dataset) * len(expressions) * 8
#sums = subspace.sum()

limits = subspace.minmax()

N = 5
print("benchmarking minmax")
expr = "subspace.minmax()"
times = timeit.repeat(expr, setup="from __main__ import subspace, dataset, np", repeat=5, number=N)
print("minimum time", min(times)/N)
bandwidth = [byte_size/1024.**3/(time/N) for time in times]
print("%f GiB/s" % max(bandwidth))

speed = [len(dataset)/(time/N)/1e9 for time in times]
print("%f billion rows/s " % max(speed))


print("benchmarking histogram")
expr = "subspace.histogram(limits, 256)"
times = timeit.repeat(expr, setup="from __main__ import subspace, dataset, np, limits", repeat=5, number=N)
print("minimum time", min(times)/N)
bandwidth = [byte_size/1024.**3/(time/N) for time in times]
print("%f GiB/s" % max(bandwidth))

speed = [len(dataset)/(time/N)/1e9 for time in times]
print("%f billion rows/s " % max(speed))

#expr = "np.nansum(dataset.columns['x'])"
print("benchmarking sum")
expr = "subspace.sum()"
times = timeit.repeat(expr, setup="from __main__ import subspace, dataset, np", repeat=5, number=N)
print("minimum time", min(times)/N)
bandwidth = [byte_size/1024.**3/(time/N) for time in times]
print("%f GiB/s" % max(bandwidth))




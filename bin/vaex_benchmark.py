from __future__ import print_function
import vaex as vx
import sys

#import yappi

#vx.set_log_level_debug()
progressbar = True
fn = sys.argv[1]
print("opening", fn)
#dataset = vx.open_many([fn])
dataset = vx.open(fn)
#dataset.set_active_fraction(0.5)

expressions = tuple(sys.argv[2:])
if sys.argv[2] == "Alpha":
	dataset.add_virtual_columns_celestial("Alpha", "Delta", "l", "b")
	expressions = ("l", "b")
	for key, value in dataset.virtual_columns.items():
		print(key, value)
#dsa
print("subspace", expressions)
subspace = dataset(*expressions)
#print "calculate minmax"
#yappi.start()
limits = subspace.minmax(progressbar=progressbar)
#print "calculate histogram"
subspace.histogram(limits, progressbar=progressbar)
#yappi.get_func_stats().print_all()


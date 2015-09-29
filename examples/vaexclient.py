__author__ = 'breddels'
import vaex as vx
import numpy as np
import pylab


server = vx.server("localhost")


list = server.list_datasets()
ds = server.open(list[5])
expr = ds("x", "y")
limits = expr.limits_sigma(sigmas=3, square=True)

ds.select("z>50")
selected = expr.selected()


print expr.mean()
print expr.var()
print expr.limits_sigma()
print expr.limits_sigma(sigmas=1)
#limits = expr.minmax()
print "square limits", limits
grid = expr.histogram(limits=limits)
grid_selected = selected.histogram(limits=limits)
expr.plot(np.log(grid), limits=limits)
pylab.contour(np.log(grid_selected), 2, linewidth="2pt", colors="blue", extent=limits.flatten(), alpha=0.8)



pylab.show()


print "datasets", list
dsa
dataset = None
import traceback

def printminmax(args):
	expr, limits = args
	print "minmax for", expr, limits
	return expr, limits

def getminmax__(dataset):
	dataset("x", "y").minmax()
	print "get min max for", dataset
	#try:
	if 1:
		expr = dataset("x", "y")
		print "expr", expr
		promise = expr.minmax()
		print "promise", promise
		#promise.then(printminmax, on_error)
		return promise
	#except Exception, e:
	#	print "issue", e
	#	raise

def list_datasets(result):
	global dataset
	import json
	list = json.loads(result.body)
	print "list datasets", list
	opened = server.open(list[5])
	def plot(args):
		expr, limits, size, data = args
		print "plot"
		#pylab.imshow(log(data))
		pylab.figure()
		print data.max(), data.sum()
		expr.plot(np.log(data), limits=limits)
		pylab.show()
		#expr.plot()
	def histogram(args):
		expr, limits = args
		print expr
		return expr.histogram(limits=limits, size=256).then(plot)
	opened.then(lambda dataset: dataset("x", "y").minmax()).then(printminmax, on_error).then(histogram).then(None, on_error)


def on_error(exception=None):
	print "error", exception
	traceback.print_exc()

end = server.list_datasets().then(list_datasets, on_error)

#end.get(10) # wait max 10 seconds
#server.wait()
#import time
#time.sleep(10)
#pylab.show()
server.wait()
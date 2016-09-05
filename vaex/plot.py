__author__ = 'breddels'
import matplotlib.pyplot as plt
import numpy as np

def grid(grid, **kwargs):
	ranges = [grid.grids.ranges[0][0], grid.grids.ranges[0][1], grid.grids.ranges[1][0], grid.grids.ranges[1][1]]
	plt.imshow(np.log(grid.data), origin="lower", extent=ranges, **kwargs)
	plt.xlabel(grid.grids.expressions[0])
	plt.ylabel(grid.grids.expressions[1])
	plt.xlim(ranges[0], ranges[1])
	plt.ylim(ranges[2], ranges[3])

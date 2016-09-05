import pylab
import numpy as np
import os
import json
import sys

name = "{name}" if len(sys.argv) == 1 else sys.argv[1]

filename_grid = name + "_grid.npy"
filename_grid_vector = name + "_grid_vector.npy"
filename_grid_selection = name + "_grid_selection.npy"
filename_meta = name + "_meta.json"

has_selection = os.path.exists(filename_grid_selection)
has_vectors = os.path.exists(filename_grid_vector)

meta = json.load(file(filename_meta))
grid = np.load(filename_grid)

xmin, xmax, ymin, ymax = meta["extent"]

if has_selection:
	grid_selection = np.load(filename_grid_selection)
	pylab.imshow(grid, extent=meta["extent"], origin="lower", cmap="binary")
	pylab.imshow(grid_selection, alpha=0.4, extent=meta["extent"], origin="lower")
else:
	pylab.imshow(grid, extent=meta["extent"], origin="lower")

if has_vectors:
	vx, vy, vz = np.load(filename_grid_vector)
	N = vx.shape[0]
	dx = (xmax-xmin)/float(N)
	dy = (ymax-ymin)/float(N)
	x = np.linspace(xmin+dx/2, xmax-dx/2, N)
	y = np.linspace(ymin+dy/2, ymax-dy/2, N)
	x2d, y2d = np.meshgrid(x, y)
	pylab.quiver(x2d, y2d, vx, vy, color="white")

pylab.show()


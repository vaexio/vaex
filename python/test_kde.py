from numpy import *

N = 100

x = random.normal(0.5, 0.015, N)
y = random.normal(0.2, 0.01, N)

import subspacefind
Nx = 50
Ny = 50
kde = subspacefind.DensityMap2d(0., 1., Nx, 0., 1., Ny);

print dir(kde)

p = x * 0
kde.comp_data_probs_2d(1., 1., x, y, p)
print p

kde.comp_density_2d(1., 1., 0.01, x, y, p)
image = zeros((Ny,Nx)) * 0.
kde.fill(image)

print image

import matplotlib.pyplot as plt
plt.imshow(image)
plt.show()


if 0:

	from kaplot import *
	box()
	indexedimage(image)
	draw()
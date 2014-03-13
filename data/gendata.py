#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
from optparse import OptionParser
import h5py

from numpy import *

def exit(msg, errorcode=1):
	print >>sys.stderr, msg
	sys.exit(errorcode)
	
def info(msg, newline=True, indent=0, *args):
	if options.verbose:
		print >>sys.stderr, "  " * indent,
		if newline:
			print >>sys.stderr, msg, " ".join(args)
		else:
			print >>sys.stderr, msg, " ".join(args),
	
parser = OptionParser()

parser.add_option("-d", "--dimension",
                  help="dimensionailty of data (number of columns) [default=%default]", default=2, type=int)
parser.add_option("-N", "--number",
                  help="number of rows[default=%default]", default=10, type=int)
parser.add_option("-s", "--sigma",
                  help="default sigma(s) [default=%default]", default="0.01", type=str)
parser.add_option("-m", "--mean",
                  help="default mean(s) [default=%default]", default="0.4", type=str)
parser.add_option("-u", "--uniform",
                  help="percentage of uniform noise [default=%default]", default="0.1", type=float)
parser.add_option("-b", "--bounds",
                  help="bounds of space for bounding box[default=%default]", default="0.0,1.0", type=str)
                  
parser.add_option("-v", "--verbose", action="store_true", default=False, help="give verbose output")
parser.add_option("-o", "--output")
(options, args) = parser.parse_args()

N = options.number
dim = options.dimension

sigma_list = []
for sigmastr in options.sigma.split(":"):
	if "," in sigmastr:
		sigmas = map(float, sigmastr.split(","))
	else:
		sigmas = [float(sigmastr)] * dim
	if len(sigmas) != dim:
		exit("length of sigmas doesn't equal number of dimensions: %s should have %d components" % (sigmastr, dim))
	sigma_list.append(sigmas)
mean_list = []
for meanstr in options.mean.split(":"):
	if "," in meanstr:
		meanss = map(float, meanstr.split(","))
	else:
		means = [float(meanstr)] * dim
	if len(means) != dim:
		exit("length of means doesn't equal number of dimensions: %s should have %d components" % (meanstr, dim))
	mean_list.append(means)
	
bounds_list = []
for bounds_str in options.bounds.split(":"):
	parts = bounds_str.split(",")
	if len(parts) != 2:
		exit("bounds exists of 2 number (upper and lower bound) for each dimension, you provided %d numbers" % len(parts))
	bound_x, bound_y = map(float, parts)
	bounds_list.append([bound_x, bound_y])
if len(bounds_list) == 1:
	bounds_list = bounds_list * dim
if len(bounds_list) != dim:
	exit("lenght of bound list should equal the number of dimensions (%d), or 1, not %d" % (dim, len(bounds_list)))
		
if len(mean_list) != len(sigma_list):
	exit("number of means provided (%d) doesn't match number of sigmas provided (%d)" % (len(mean_list), len(sigma_list)))

parts = len(mean_list)
if (N % parts) != 0:
	exit("requested %d rows, but is not a multiple of %d (number of gaussians requested)" % (N, parts))

#sigmas = [float(sigma.strip()) for sigma in options.sigma.split(":")]
#means = [float(mean.strip()) for mean in options.mean.split(":")]
info("sigmas: %r", sigma_list)
info("means: %r", mean_list)
#maxcount 


if options.output:
	h5output = h5py.File(options.output, "w", driver="core")
	dataset = h5output.create_dataset("data", (dim,N), dtype='f64')
else:
	h5output = None
	
index = 0

def output(row):
	global index
	if h5output:
		dataset[:,index] = array(row)
	else:
		print " ".join(map(str, row))
	index += 1
	
	
random.seed(1)
for part in range(parts):
	info("part %d out of %d" % (part+1, parts))
	Ngaussian = int(N/parts * (1-options.uniform))
	Nuniform = N/parts - Ngaussian
	Ngaussian_total = 0
	Ngaussian_sample = min(10000, max(Ngaussian/4, 100))
	while Ngaussian_total < Ngaussian:
		gaussian = zeros((Ngaussian_sample,dim))
		mask = gaussian[:,0] == 0 # initial value for mask is all true
		for d in range(dim):
			gaussian[:,d] = random.normal(mean_list[part][d], sigma_list[part][d], Ngaussian_sample)
			mask = mask & ((gaussian[:,d] >= bounds_list[d][0]) & (gaussian[:,d] < bounds_list[d][1]))
		info("number of gaussians sampled: %d" % len(gaussian), indent=1)
		gaussian = gaussian[mask]
		Ngaussian_todo = Ngaussian - Ngaussian_total
		if len(gaussian) > Ngaussian_todo:
			info("more sampled than needed (%d), needed only %d" % (len(gaussian), Ngaussian_todo), indent=1)
			gaussian = gaussian[:Ngaussian_todo]
		Ngaussian_total += len(gaussian)
		info("leftover after clipping: %d" % len(gaussian), indent=1)
		info("%d of %d sampled" % (Ngaussian_total, Ngaussian), indent=1)
		#for i in range(len(gaussian)):
		#	output(gaussian[i])
		map(output, gaussian)
	
	Nuniform_total = 0
	Nuniform_sample = min(10000, max(Nuniform/4, 100))
	while Nuniform_total < Nuniform:
		uniform = zeros((Nuniform_sample,dim))
		mask = uniform[:,0] == 0 # initial value for mask is all true
		for d in range(dim):
			uniform[:,d] = random.random(Nuniform_sample) * (bounds_list[d][1] - bounds_list[d][0]) + bounds_list[d][0]
		info("number of uniforms sampled: %d" % len(uniform), indent=1)
		Nuniform_todo = Nuniform - Nuniform_total
		if len(uniform) > Nuniform_todo:
			info("more sampled than needed (%d), needed only %d" % (len(uniform), Nuniform_todo), indent=1)
			uniform = uniform[:Nuniform_todo]
		Nuniform_total += len(uniform)
		info("leftover after clipping: %d" % len(uniform), indent=1)
		info("%d of %d sampled" % (Nuniform_total, Nuniform), indent=1)
		#for i in range(len(uniform)):
		#	output(uniform[i])
		map(output, uniform)
	
if 0:
	x = random.normal(0.5, 0.015, N/2)
	y = random.normal(0.2, 0.01, N/2)

	for i in range(N/2):
		print x[i], y[i]
	x = random.normal(0.6, 0.015, N/2)
	y = random.normal(0.7, 0.1, N/2)

	for i in range(N/2):
		print x[i], y[i]

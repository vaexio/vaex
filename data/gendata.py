#!/usr/bin/env python
# -*- coding: utf-8 -*-
from numpy import *
N = 100000000
x = random.normal(0.5, 0.015, N/2)
y = random.normal(0.2, 0.01, N/2)

for i in range(N/2):
	print x[i], y[i]
x = random.normal(0.6, 0.015, N/2)
y = random.normal(0.7, 0.1, N/2)

for i in range(N/2):
	print x[i], y[i]

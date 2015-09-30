# -*- coding: utf-8 -*-
__author__ = 'maartenbreddels'

import unittest
import numpy as np
import vaex.vaexfast
import numpy.testing

class MyTestCase(unittest.TestCase):

	def test_histogram1d(self):

		x = np.arange(-5-0.1, 5-0.1, dtype=np.float64)
		print x, sum(x)
		for N in [1, 2, 4, 256, 512]:
			counts = np.zeros(N, dtype=np.float64)
			min, max = -5, 5
			vaex.vaexfast.histogram1d(x, None, counts, min, max) #+1e-15)
			#print np.sum(counts), len(x), x
			self.assertEqual(np.sum(counts), 9, "histogram1d test") # 1 should fall outside

			counts = np.zeros((N, N), dtype=np.float64)
			vaex.vaexfast.histogram2d(x, x, None, counts, min, max, min, max)
			self.assertEqual(np.sum(counts), 9, "histogram2d test") # 1 should fall outside

			counts = np.zeros((N, N, N), dtype=np.float64)
			vaex.vaexfast.histogram3d(x, x, x, None, counts, min, max, min, max, min, max)
			self.assertEqual(np.sum(counts), 9, "histogram3d test") # 1 should fall outside
			if 0:
				print np.sum(counts)

				counts = np.zeros((N, N, N), dtype=np.float64)
				vaex.vaexfast.histogram3d(x, x, x, None, counts, 0., 9., 0., 9., 0., 9.)
				print np.sum(counts)
	def _test_resize(self):
		if 1:
			input1d = np.arange(4) * 1.
			output1d = vaex.vaexfast.resize(input1d, 2)
			#print input1d, output1d
			#self.assert_(np.array_equal(output1d, np.array([1., 5.])))
			np.testing.assert_equal(np.array([1., 5.]), output1d)


		input2d = (np.arange(4) * 1.).reshape(2,2)
		output2d = vaex.vaexfast.resize(input2d, 1)
		#self.assertEqual(output2d, np.array([[6.]]))
		np.testing.assert_equal(np.array([[6.]]), output2d)

		input2d = (np.arange(16) * 1.).reshape(4,4)
		"""
[[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]]"""
		output2d = vaex.vaexfast.resize(input2d, 2)
		#self.assertEqual(output2d, np.array([[6.]]))
		np.testing.assert_equal(np.array([[0.+1+4+5, 2+3+6+7], [8+9+12+13, 10+11+14+15] ]), output2d)


		N = 256
		input2d = (np.arange(N**2) * 1.).reshape(N,N)
		while True:
			N = N // 8
			#print N
			if N == 0:
				break
			output2d = vaex.vaexfast.resize(input2d, N)
			for i in range(3):
				input2d = input2d[::2,::] + input2d[1::2,::]
				input2d = input2d[::,::2] + input2d[::,1::2]
			if 0:
				print "input"
				print input2d
				print "output"
				print output2d
			np.testing.assert_equal(input2d, output2d)
			np.testing.assert_equal(np.sum(input2d), np.sum(output2d))


		if 1:
			for N in [8, 16, 32, 64, 128, 256]:
				input2d = (np.arange(N**2) * 1.).reshape(N,N)
				input3d = (np.arange(N**3) * 1.).reshape(N,N,N)
				total2d = np.sum(input2d)
				total3d = np.sum(input3d)
				for N2 in [1, 2, 4, 8, 16, 32, 64]:
					if N2 <= N:
						output2d = vaex.vaexfast.resize(input2d, N2)
						output3d = vaex.vaexfast.resize(input3d, N2)
						#print output2d, total
						np.testing.assert_equal(np.sum(output2d), total2d)
						np.testing.assert_equal(np.sum(output3d), total3d)


if __name__ == '__main__':
	unittest.main()

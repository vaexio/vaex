__author__ = 'maartenbreddels'

import unittest
import numpy as np
import gavifast
import numpy.testing

class MyTestCase(unittest.TestCase):
	def test_resize(self):
		if 1:
			input1d = np.arange(4) * 1.
			output1d = gavifast.resize(input1d, 2)
			#print input1d, output1d
			#self.assert_(np.array_equal(output1d, np.array([1., 5.])))
			np.testing.assert_equal(np.array([1., 5.]), output1d)


		input2d = (np.arange(4) * 1.).reshape(2,2)
		output2d = gavifast.resize(input2d, 1)
		#self.assertEqual(output2d, np.array([[6.]]))
		np.testing.assert_equal(np.array([[6.]]), output2d)

		input2d = (np.arange(16) * 1.).reshape(4,4)
		"""
[[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  ¡™14.  15.]]"""
		output2d = gavifast.resize(input2d, 2)
		#self.assertEqual(output2d, np.array([[6.]]))
		np.testing.assert_equal(np.array([[0.+1+4+5, 2+3+6+7], [8+9+12+13, 10+11+14+15] ]), output2d)

if __name__ == '__main__':
	unittest.main()

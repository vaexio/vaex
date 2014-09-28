import gavi.dataset as dataset
import numpy as np
import unittest

class DatasetTest(unittest.TestCase):
	def setUp(self):
		self.dataset = dataset.MemoryMapped("test", nommap=True)

		x = np.arange(10)
		y = x ** 2
		self.dataset.addColumn("x", array=x)
		self.dataset.addColumn("y", array=y)
		
		self.jobsManager = dataset.JobsManager()
		
	def length_test(self):
		assert len(self.dataset) == 10

	def length_mask_test(self):
		self.dataset.selectMask(self.dataset.columns['x'] < 5)
		self.assertEqual(self.dataset.length(selection=True), 5)
		
	

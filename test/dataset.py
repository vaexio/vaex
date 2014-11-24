import gavi.dataset as dataset
import numpy as np
import unittest

class TestDataset(unittest.TestCase):
	def setUp(self):
		self.dataset = dataset.MemoryMapped("test", nommap=True)

		x = np.arange(10)
		y = x ** 2
		self.dataset.addColumn("x", array=x)
		self.dataset.addColumn("y", array=y)
		
		self.jobsManager = dataset.JobsManager()
		
	def test_length(self):
		assert len(self.dataset) == 10

	def test_length_mask(self):
		self.dataset.selectMask(self.dataset.columns['x'] < 5)
		self.assertEqual(self.dataset.length(selection=True), 5)
		
	
if __name__ == '__main__':
    unittest.main()
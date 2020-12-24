__author__ = 'kmcentush'

import unittest
import numpy as np
import pyarrow as pa
from vaex.column import _to_string_sequence
from vaex.superstrings import StringArray


class TestNumpyArrayToStringSequence(unittest.TestCase):
    def test_numpy_array_to_str_sequence(self):
        np_array = np.array([1.1, -2.4, 3.2], dtype='float64')
        pa_array = pa.array(np_array)

        self.assertEqual(len(pa_array.buffers()), 2)
        self.assertIsInstance(_to_string_sequence(pa_array), StringArray)


if __name__ == '__main__':
    unittest.main()

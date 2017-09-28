__author__ = 'breddels'

import unittest
import vaex as vx
import vaex.utils
import vaex.image
import numpy as np
default_size = 2
default_shape = (default_size, default_size)

class TestImage(unittest.TestCase):
    def test_blend(self):
        black = vaex.image.background(default_shape, "black")
        white = vaex.image.background(default_shape, "white")
        transparant_black = black * 1
        transparant_black[...,3] = 100
        for mode in vaex.image.modes:
            self.assert_(np.all(vaex.image.blend([black, white]) == white))
            self.assert_(np.all(vaex.image.blend([white, black]) == black))

            grey = vaex.image.blend([white, transparant_black], mode)

            self.assert_(np.all(grey[...,0:3] < white[...,0:3]))
            self.assert_(np.all(grey[...,0:3] > black[...,0:3]))

    def test_pil(self):
        white = vaex.image.background(default_shape, "white")
        # not much to test
        im = vaex.image.rgba_2_pil(white)
        raw_data = vaex.image.pil_2_data(im)
        url = vaex.image.rgba_to_url(white)


class TestAttrDict(unittest.TestCase):
    def test_attrdict(self):
        d = vx.utils.AttrDict()
        d.a = 1
        self.assertEqual(d.a, 1)
        with self.assertRaises(KeyError):
            a = d.doesnotexist


        d = vx.utils.AttrDict(a=1)
        print(d.__dict__)
        self.assertEqual(d.a, 1)
        with self.assertRaises(KeyError):
            a = d.doesnotexist

if __name__ == '__main__':
    unittest.main()


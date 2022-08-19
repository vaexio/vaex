import glob

basedir = 'tests/data/images'

import vaex.vision
import PIL

def test_image_open():
    df = vaex.vision.open(basedir)
    assert isinstance(df.image.tolist()[0], PIL.Image.Image)
    assert df.image.vision.as_numpy().shape == (len(df), 261, 350, 3)
    assert df.image.vision.resize((8, 4)).vision.as_numpy().shape == (2, 4, 8, 4)

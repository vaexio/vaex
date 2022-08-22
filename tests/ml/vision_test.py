import vaex.vision
import PIL

basedir = 'tests/data/images'


def test_image_open():
    df = vaex.vision.open(basedir)
    assert df.shape == (16, 2)
    assert isinstance(df.image.tolist()[0], PIL.Image.Image)
    assert df.image.vision.to_numpy().shape == (16, 261, 350, 3)
    assert df.image.vision.resize((8, 4)).vision.to_numpy().shape == (16, 4, 8, 3)

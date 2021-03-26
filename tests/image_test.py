import os
import vaex
import vaex.image
import PIL.Image

path = os.path.dirname(__file__)
image_path = os.path.join(path, 'data', 'basn6a16.png')


def test_image_open():
    df = vaex.from_arrays(path=[image_path, image_path])
    df['img'] = df.path.image.open()
    assert isinstance(df.img.tolist()[0], PIL.Image.Image)
    assert df.img.image.as_numpy().shape == (2, 32, 32, 4)
    assert df.img.image.resize((8, 4)).image.as_numpy().shape == (2, 4, 8, 4)

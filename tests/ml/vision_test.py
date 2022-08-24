import vaex.vision
import PIL

basedir = 'tests/data/images'


def test_vision_conversions():
    df = vaex.vision.open(basedir)
    df['image_bytes'] = df['image'].vision.to_bytes()
    df['image_str'] = df['image'].vision.to_str()
    df['image_array'] = df['image'].vision.resize((10, 10)).vision.to_numpy()

    assert isinstance(df['image_bytes'].vision.from_bytes().values[0], PIL.Image.Image)
    assert isinstance(df['image_str'].vision.from_str().values[0], PIL.Image.Image)
    assert isinstance(df['image_array'].vision.from_numpy().values[0], PIL.Image.Image)

    assert isinstance(df['image_bytes'].vision.infer().values[0], PIL.Image.Image)
    assert isinstance(df['image_str'].vision.infer().values[0], PIL.Image.Image)
    assert isinstance(df['image_array'].vision.infer().values[0], PIL.Image.Image)
    assert isinstance(df['path'].vision.infer().values[0], PIL.Image.Image)


def test_vision_open():
    df = vaex.vision.open(basedir)
    assert df.shape == (4, 2)
    assert vaex.vision.open(basedir + '/dogs').shape == (2, 2)
    assert vaex.vision.open(basedir + '/dogs/dog*').shape == (2, 2)
    assert vaex.vision.open(basedir + '/dogs/dog.2423.jpg').shape == (1, 2)
    assert vaex.vision.open([basedir + '/dogs/dog.2423.jpg', basedir + '/cats/cat.4865.jpg']).shape == (2, 2)
    assert 'path' in df
    assert 'image' in df


def test_vision():
    df = vaex.vision.open(basedir)
    assert df.shape == (4, 2)
    assert isinstance(df.image.tolist()[0], PIL.Image.Image)
    assert df.image.vision.to_numpy().shape == (4, 261, 350, 3)
    assert df.image.vision.resize((8, 4)).vision.to_numpy().shape == (4, 4, 8, 3)

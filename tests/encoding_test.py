import vaex.encoding
import numpy as np
import pyarrow as pa

@vaex.encoding.register('blobtest')
class encoding:
    @staticmethod
    def encode(encoding, data):
        return {'someblob': encoding.add_blob(data['someblob'])}

    @staticmethod
    def decode(encoding, data):
        return {'someblob': encoding.get_blob(data['someblob'])}


def test_encoding():
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('blobtest', {'someblob': b'1234'})
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    values = encoding.decode('blobtest', data)
    assert values['someblob'] == b'1234'


def test_encoding_arrow():
    x = pa.array(np.arange(10, dtype='f4'))
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('arrow-array', x)
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    value = encoding.decode('arrow-array', data)
    assert value.to_pylist() == x.to_pylist()


def test_encoding_numpy():
    x = np.arange(10, dtype='>f4')
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('ndarray', x)
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    value = encoding.decode('ndarray', data)
    assert np.all(value == x)


def test_encoding_numpy_masked():
    x = np.arange(10, dtype='>f4')
    mask = x > 4
    x = np.ma.array(x, mask=mask)
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('ndarray', x)
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    value = encoding.decode('ndarray', data)
    assert np.all(value == x)
    assert np.all(value.mask == x.mask)


def test_encoding_numpy_datetime():
    x = np.arange('2001', '2005', dtype='M')
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('ndarray', x)
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    value = encoding.decode('ndarray', data)
    assert np.all(value == x)


def test_encoding_numpy_timedelta():
    x = np.arange('2001', '2005', dtype='M')
    x = x - x[0]
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('ndarray', x)
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    value = encoding.decode('ndarray', data)
    assert np.all(value == x)


def test_encoding_numpy_string_objects():
    x = np.array(['vaex', 'is', None, 'fast'])
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('ndarray', x)
    wiredata = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(wiredata, encoding)
    value = encoding.decode('ndarray', data)
    assert np.all(value == x)

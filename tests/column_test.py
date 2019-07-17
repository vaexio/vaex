import vaex
import numpy as np

def test_vrange():
    N = 1000**3
    df = vaex.from_arrays(x=vaex.vrange(0, N))
    assert len(df.columns['x']) == N
    trimmed = df.columns['x'].trim(2,4)
    assert trimmed.start == 2
    assert trimmed.stop == 4
    assert len(df) == N
    assert len(df[0:10]) == 10
    assert df[1:11].x.tolist() == (np.arange(1, 11.)).tolist()
    df['y'] = df.x**2
    assert df[1:11].y.tolist()== (np.arange(1, 11)**2).tolist()

def test_arrow_strings():
    N = 4
    x = ['a', 'bb', 'ccc', 'dddd']
    xc = vaex.string_column(x)
    df = vaex.from_arrays(x=xc)
    assert len(df.columns['x']) == 4
    trimmed = df.columns['x'][2:4]
    assert trimmed[:].tolist() == x[2:4]
    assert trimmed[1:2].tolist() == x[3:4]
    assert len(df) == N
    assert len(df[1:3]) == 2
    assert df[1:3].x.tolist() == x[1:3]


def test_arrow_strings_null():
    N = 4
    x = ['a', 'bb', None, 'dddd', None]
    xc = vaex.string_column(x)
    assert xc.tolist() == x
    assert xc[1:].tolist() == x[1:]
    assert xc[2:4].tolist() == x[2:4]


def test_plain_strings():
    N = 4
    x = np.array(['a', 'bb', 'ccc', 'dddd'], dtype='object')
    df = vaex.from_arrays(x=x)

    assert len(df.columns['x']) == 4
    trimmed = df.columns['x'][2:4]
    assert trimmed[:].tolist() == x[2:4].tolist()
    assert len(df) == N
    assert len(df[1:3]) == 2
    assert df[1:3].x.tolist() == x[1:3].tolist()


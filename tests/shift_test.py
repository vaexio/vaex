import collections
import pytest

import numpy as np
import pyarrow as pa
import vaex.shift


def chunk_iter(chunks, chunk_size):
    some_value = list(chunks.values())[0]
    for i in range((len(some_value) + chunk_size-1)//chunk_size):
        i1 = i*chunk_size
        i2 = min(len(some_value), (i+1)*chunk_size)
        yield i1, i2, {name: chunks[name].slice(i1, i2-i1) for name in chunks}


def eat_chunks(iter):
    offsets = []
    flat_chunks = collections.defaultdict(list)
    for i1, i2, chunks in iter:
        print(i1, i2, chunks)
        offsets.append((i1, i2))
        for name, values in chunks.items():
            flat_chunks[name].extend(vaex.array_types.tolist(values))
    return offsets, flat_chunks


def test_chunk_prepend(chunk_size=2):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    xp = pa.array([99, 88])
    xexpected = pa.array([99, 88, 0, 1, 2])
    i = chunk_iter(dict(x=x, y=y), chunk_size)
    offsets, chunks = eat_chunks(vaex.shift.chunk_prepend(i, {'x': xp}, chunk_size))
    assert offsets == [(0, 2), (2, 4), (4, 5)]
    assert chunks['x'] == xexpected.tolist()


def test_chunk_append(chunk_size=2):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    xappend = pa.array([99, 88])
    xexpected = pa.array([2, None, 4, 99, 88])
    i = chunk_iter(dict(x=x, y=y), chunk_size)
    offsets, chunks = eat_chunks(vaex.shift.chunk_append(i, {'x': xappend}, chunk_size))
    # assert offsets == [(0, 2), (2, 4), (4, 5)] TODO FIX
    assert chunks['x'] == xexpected.tolist()


def test_chunk_eat(chunk_size=2):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    i = chunk_iter(dict(x=x, y=y), chunk_size)
    offsets, chunks = eat_chunks(vaex.shift.chunk_eat(i, 2))
    assert chunks['x'] == x[2:].tolist()
    assert chunks['y'] == y[2:].tolist()
    assert offsets == [(0, 2), (2, 3)]


@pytest.mark.parametrize("length", list(range(1, 5)))
def test_chunk_trim(length, chunk_size=2):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    i = chunk_iter(dict(x=x, y=y), chunk_size)
    offsets, chunks = eat_chunks(vaex.shift.chunk_trim(i, length))
    assert chunks['x'] == x[:length].tolist()
    assert chunks['y'] == y[:length].tolist()
    assert len(chunks['x']) == length
    assert offsets[0] == (0, min(2, length))
    if len(offsets) > 1:
        assert offsets[1] == (2, min(4, length))


def test_sliding_matrix(chunk_size=2):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    xappend = pa.array([99, 88])
    xexpected = np.array([[0, 1], [1, 2], [2, -1], [-1, 4], [4, -1]])
    xexpected = np.ma.array(xexpected, mask=xexpected==-1)
    xexpected3 = np.array([[0, 1, 2], [1, 2, -1], [2, -1, 4], [-1, 4, -1], [4, -1, -1]])
    xexpected3 = np.ma.array(xexpected3, mask=xexpected3==-1)
    xresult = vaex.shift.sliding_matrix(x, None, 2)
    assert xresult.tolist() == xexpected.tolist()

    xresult3 = vaex.shift.sliding_matrix(x, None, 3)
    assert xresult3.tolist() == xexpected3.tolist()

    # xresult = vaex.shift.sliding_matrix(x[:2], x[2:], 2)
    # assert xresult.tolist() == xexpected[:2].tolist()

    xresult3 = vaex.shift.sliding_matrix(x[:2], x[2:], 3)
    assert xresult3.tolist() == xexpected3[:2].tolist()



@pytest.mark.parametrize("virtual", [False, True])
def test_shift_basics(df_factory, virtual):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    if virtual:
        df['x'] = df.x + 0
    dfp1 = df._shift(1, ['x'])
    dfn1 = df._shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2, None]
    assert dfp1.y.tolist() == [0, 1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4, None]
    assert dfn1.y.tolist() == [0, 1, None, 9, 16]

    assert dfp1._shift(1).x.tolist() == [None, None, 0, 1, 2]
    assert dfp1._shift(-1).x.tolist() == [0, 1, 2, None, None]
    assert dfp1._shift(-1, fill_value=99).x.tolist() == [0, 1, 2, None, 99]

    assert dfn1._shift(1).x.tolist() == [None, 1, 2, None, 4]
    assert dfn1._shift(-1).x.tolist() == [2, None, 4, None, None]
    assert dfn1._shift(-1, fill_value=99).x.tolist() == [2, None, 4, None, 99]

    assert df._shift(4).x.tolist() == [None, None, None, None, 0]
    assert df._shift(5).x.tolist() == [None, None, None, None, None]
    assert df._shift(6).x.tolist() == [None, None, None, None, None]

    assert df._shift(-4).x.tolist() == [4, None, None, None, None]
    assert df._shift(-5).x.tolist() == [None, None, None, None, None]
    assert df._shift(-6).x.tolist() == [None, None, None, None, None]


@pytest.mark.parametrize("length", list(range(1, 3)))
@pytest.mark.parametrize("i1", list(range(1, 3)))
def test_shift_slice(df_factory, i1, length):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    dfp1 = df._shift(1, ['x'])
    dfn1 = df._shift(-1, ['x'])
    i2 = i1 + length + 1
    assert dfp1[i1:i2].x.tolist() == [None, 0, 1, 2, None][i1:i2]
    assert dfp1[i1:i2].y.tolist() == [0, 1, None, 9, 16][i1:i2]
    assert dfn1[i1:i2].x.tolist() == [1, 2, None, 4, None][i1:i2]

def test_shift_basics_trim(df_factory):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    dfp1 = df._shift(1, ['x'], trim=True)
    dfn1 = df._shift(-1, ['x'], trim=True)
    assert dfp1.x.tolist() == [0, 1, 2, None]
    assert dfp1.y.tolist() == [1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4]
    assert dfn1.y.tolist() == [0, 1, None, 9]

    assert dfp1._shift(1, trim=True).x.tolist() == [0, 1, 2]
    assert dfp1._shift(-1, trim=True).x.tolist() == [1, 2, None]


def test_shift_range(df_factory):
    x = [0, 1, 2, 3, 4]
    xm1 = [1, 2, 3, 4, None]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    df['x1'] = df['x']
    df['x2'] = df['x']
    df._shift(0, ['x1'], inplace=True)
    df._shift(-1, ['x2'], inplace=True)
    assert df.x1.tolist() == x
    assert df.x2.tolist() == xm1
    assert df.func.stack([df.x1, df.x2]).tolist() == [[0, 1], [1, 2], [2, 3], [3, 4], [4, None]]
    df = df_factory(x=x, y=y)
    df._shift((0, 2), 'x', inplace=True)
    assert df.x.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4], [4, None]]

    # trim with range
    df = df_factory(x=x, y=y)
    df._shift((0, 3), 'x', inplace=True, trim=True)
    assert df.x.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]



def test_shift_filtered(df_factory):
    x = [0, 99, 1, 99, 2, 99, None, 99, 4, 99]
    y = [0, 88, 1, 88, None, 88, 9, 88, 16, 88]
    assert len(x) == len(y)
    df = df0 = df_factory(x=x, y=y)
    df = df[((df.x != 99) | df.x.ismissing()).fillna(True)]
    dfp1 = df._shift(1, ['x'])
    dfn1 = df._shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2, None]
    assert dfp1.y.tolist() == [0, 1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4, None]
    assert dfn1.y.tolist() == [0, 1, None, 9, 16]

    assert dfp1._shift(1).x.tolist() == [None, None, 0, 1, 2]
    assert dfp1._shift(-1).x.tolist() == [0, 1, 2, None, None]
    assert dfp1._shift(-1, fill_value=99).x.tolist() == [0, 1, 2, None, 99]

    assert dfn1._shift(1).x.tolist() == [None, 1, 2, None, 4]
    assert dfn1._shift(-1).x.tolist() == [2, None, 4, None, None]
    assert dfn1._shift(-1, fill_value=99).x.tolist() == [2, None, 4, None, 99]

    assert df._shift(4).x.tolist() == [None, None, None, None, 0]
    assert df._shift(5).x.tolist() == [None, None, None, None, None]
    assert df._shift(6).x.tolist() == [None, None, None, None, None]

    assert df._shift(-4).x.tolist() == [4, None, None, None, None]
    assert df._shift(-5).x.tolist() == [None, None, None, None, None]
    assert df._shift(-6).x.tolist() == [None, None, None, None, None]


def test_shift_string(df_factory):
    x = np.arange(4)
    s = pa.array(['aap', None, 'noot', 'mies'])
    df = df_factory(x=x, s=s)
    assert df._shift(1).s.tolist() == [None, 'aap', None, 'noot']
    assert df._shift(-1).s.tolist() == [None, 'noot', 'mies', None]
    assert df._shift(1, ['s'], fill_value='VAEX').s.tolist() == ['VAEX', 'aap', None, 'noot']
    assert df._shift(-1, ['s'], fill_value='VAEX').s.tolist() == [None, 'noot', 'mies', 'VAEX']


def test_shift_virtual(df_factory):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    xsp1 = [None, 0, 1, 2, None]
    xsn1 = [1, 2, None, 4, None]
    df = df_factory(x=x, y=y)

    # # a is a virtual column that depends on x, but we don't shift a
    df['a'] = df.x + 0
    df['b'] = df.a
    dfs = df._shift(1, ['x'])
    assert dfs.x.tolist() == xsp1
    assert dfs.a.tolist() == x
    assert dfs.y.tolist() == y
    dfs = df._shift(-1, ['x'])
    assert dfs.x.tolist() == xsn1
    assert dfs.a.tolist() == x
    assert dfs.y.tolist() == y

    # a is a virtual column that depends on x, we shift a, but we don't shift x
    # we expect, a: __x_shifted, x: __x
    df = df_factory(x=x, y=y)
    df['a'] = df.x + 0
    dfs = df._shift(1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsp1
    assert dfs.y.tolist() == y
    dfs = df._shift(-1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsn1
    assert dfs.y.tolist() == y

    # same, but now we also have a reference to a, which we also do not shift
    df = df_factory(x=x, y=y)
    df['a'] = df.x + 0
    df['b'] = df.a + 0
    dfs = df._shift(1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsp1
    assert dfs.b.tolist() == x
    assert dfs.y.tolist() == y
    dfs = df._shift(-1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsn1
    assert dfs.b.tolist() == x
    assert dfs.y.tolist() == y


def test_shift_dataset(chunk_size=2):
    x = np.arange(5)
    y = x**2
    ds = vaex.dataset.DatasetArrays(x=x, y=y)
    dss = vaex.shift.DatasetShifted(ds, column_mapping={'x': 'x_shift'}, start=0, end=0)
    offsets, chunks = eat_chunks(dss.chunk_iterator({'x'}, chunk_size=chunk_size))
    assert chunks == {'x': x.tolist()}

    offsets, chunks = eat_chunks(dss.chunk_iterator({'x_shift'}, chunk_size=chunk_size))
    assert chunks == {'x_shift': x.tolist()}

    offsets, chunks = eat_chunks(dss.chunk_iterator({'x_shift', 'x'}, chunk_size=chunk_size))
    assert chunks == {'x': x.tolist(), 'x_shift': x.tolist()}

    xs = [None] + x[:-1].tolist()

    dss = vaex.shift.DatasetShifted(ds, column_mapping={'x': 'x_shift'}, start=1, end=1)

    offsets, chunks = eat_chunks(dss.chunk_iterator({'x'}, chunk_size=chunk_size))
    assert chunks == {'x': x.tolist()}

    offsets, chunks = eat_chunks(dss.chunk_iterator({'x_shift'}, chunk_size=chunk_size))
    assert chunks == {'x_shift': xs}

    offsets, chunks = eat_chunks(dss.chunk_iterator({'x_shift', 'x', 'y'}, chunk_size=chunk_size))
    assert chunks == {'x': x.tolist(), 'x_shift': xs, 'y': y.tolist()}


    # two columns shifted
    dss = vaex.shift.DatasetShifted(ds, column_mapping={'x': 'x_shift', 'y': 'y_shift'}, start=1, end=1)
    dss_range = vaex.shift.DatasetShifted(ds, column_mapping={'x': 'x_shift', 'y': 'y_shift'}, start=1, end=2)
    offsets, chunks = eat_chunks(dss.chunk_iterator({'x_shift'}, chunk_size=chunk_size))
    assert chunks == {'x_shift': xs}

    assert dss.shape('x_shift') == dss.shape('x')

    assert not dss.is_masked('x_shift')
    assert dss_range.is_masked('x_shift')

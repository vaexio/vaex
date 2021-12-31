import collections
import pytest

import numpy as np
import pyarrow as pa
import vaex.shift
import vaex.ml


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
    offsets, chunks = eat_chunks(vaex.shift.chunk_eat(i, 3))
    assert chunks['x'] == x[3:].tolist()
    assert chunks['y'] == y[3:].tolist()
    assert offsets == [(0, 1), (1, 2)]


@pytest.mark.parametrize("length", list(range(1, 5)))
@pytest.mark.parametrize("chunk_size", [2, 5, 10])
def test_chunk_trim(length, chunk_size):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    i = chunk_iter(dict(x=x, y=y), chunk_size)
    offsets, chunks = eat_chunks(vaex.shift.chunk_trim(i, length))
    assert chunks['x'] == x[:length].tolist()
    assert chunks['y'] == y[:length].tolist()
    assert len(chunks['x']) == length
    assert offsets[0] == (0, min(chunk_size, length))
    if len(offsets) > 1:
        assert offsets[1] == (2, min(4, length))


def test_sliding_matrix(chunk_size=2):
    x = pa.array([0, 1, 2, None, 4])
    y = pa.array([0, 1, None, 9, 16])
    xappend = pa.array([99, 88])
    xexpected = np.array([[0, 1], [1, 2], [2, -1], [-1, 4], [4, -1]])
    xexpected = np.ma.array(xexpected, mask=xexpected==-1)

    xresult = vaex.shift.sliding_matrix(None, x, None, 2, 0)
    assert xresult.tolist() == xexpected.tolist()

    xresult = vaex.shift.sliding_matrix(x[:2], x[2:4], x[4:], 2, 0)
    assert xresult.tolist() == xexpected[2:4].tolist()

    # with 3 elements
    xexpected3 = np.array([[0, 1, 2], [1, 2, -1], [2, -1, 4], [-1, 4, -1], [4, -1, -1]])
    xexpected3 = np.ma.array(xexpected3, mask=xexpected3==-1)

    xresult3 = vaex.shift.sliding_matrix(None, x, None, 3, 0)
    assert xresult3.tolist() == xexpected3.tolist()

    xresult3 = vaex.shift.sliding_matrix(x[:2], x[2:4], x[4:], 3, 0)
    assert xresult3.tolist() == xexpected3[2:4].tolist()


    # using offset
    xexpected = np.array([[-1, 0], [0, 1], [1, 2], [2, -1], [-1, 4]])
    xexpected = np.ma.array(xexpected, mask=xexpected==-1)

    xresult = vaex.shift.sliding_matrix(None, x, None, 2, 1)
    assert xresult.tolist() == xexpected.tolist()

    xresult = vaex.shift.sliding_matrix(x[:2], x[2:4], x[4:], 2, 1)
    assert xresult.tolist() == xexpected[2:4].tolist()

    # offset 2
    xexpected = np.array([[-1, -1], [-1, 0], [0, 1], [1, 2], [2, -1]])
    xexpected = np.ma.array(xexpected, mask=xexpected==-1)

    xresult = vaex.shift.sliding_matrix(None, x, None, 2, 2)
    assert xresult.tolist() == xexpected.tolist()

    xresult = vaex.shift.sliding_matrix(x[:2], x[2:4], x[4:], 2, 2)
    assert xresult.tolist() == xexpected[2:4].tolist()

    # offset 1 and 3 elements
    xexpected3 = np.array([[None, 0, 1], [0, 1, 2], [1, 2, -1], [2, -1, 4], [-1, 4, -1]])
    xexpected3 = np.ma.array(xexpected3, mask=xexpected3==-1)

    xresult3 = vaex.shift.sliding_matrix(None, x, None, 3, 1)
    assert xresult3.tolist() == xexpected3.tolist()

    xresult3 = vaex.shift.sliding_matrix(x[:2], x[2:4], x[4:], 3, 1)
    assert xresult3.tolist() == xexpected3[2:4].tolist()

    # offset 2 and 3 elements
    xexpected3 = np.array([[None, None, 0], [None, 0, 1], [0, 1, 2], [1, 2, -1], [2, -1, 4]])
    xexpected3 = np.ma.array(xexpected3, mask=xexpected3==-1)

    xresult3 = vaex.shift.sliding_matrix(None, x, None, 3, 2)
    assert xresult3.tolist() == xexpected3.tolist()

    xresult3 = vaex.shift.sliding_matrix(x[:2], x[2:4], x[4:], 3, 2)
    assert xresult3.tolist() == xexpected3[2:4].tolist()


@pytest.mark.parametrize("virtual", [False, True])
def test_shift_basics(df_factory, virtual, rebuild_dataset):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    if virtual:
        df['x'] = df.x + 0
    dfp1 = df.shift(1, ['x'])
    dfn1 = df.shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2, None]
    assert dfp1.y.tolist() == [0, 1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4, None]
    assert dfn1.y.tolist() == [0, 1, None, 9, 16]

    assert dfp1.shift(1).x.tolist() == [None, None, 0, 1, 2]
    assert dfp1.shift(-1).x.tolist() == [0, 1, 2, None, None]
    assert dfp1.shift(-1, fill_value=99).x.tolist() == [0, 1, 2, None, 99]

    assert dfn1.shift(1).x.tolist() == [None, 1, 2, None, 4]
    assert dfn1.shift(-1).x.tolist() == [2, None, 4, None, None]
    assert dfn1.shift(-1, fill_value=99).x.tolist() == [2, None, 4, None, 99]

    assert df.shift(4).x.tolist() == [None, None, None, None, 0]
    assert df.shift(5).x.tolist() == [None, None, None, None, None]
    assert df.shift(6).x.tolist() == [None, None, None, None, None]

    assert df.shift(-4).x.tolist() == [4, None, None, None, None]
    assert df.shift(-5).x.tolist() == [None, None, None, None, None]
    assert df.shift(-6).x.tolist() == [None, None, None, None, None]

    dfp1_rebuild = vaex.from_dataset(rebuild_dataset(dfp1.dataset))
    dfp1_rebuild.state_set(dfp1.state_get())
    assert dfp1_rebuild.x.tolist() == dfp1.x.tolist()
    # assert rebuild_dataset(df.shift(1).hashed()) == df.shift(1).hashed()


@pytest.mark.parametrize("length", list(range(1, 3)))
@pytest.mark.parametrize("i1", list(range(1, 3)))
def test_shift_slice(df_factory, i1, length):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    dfp1 = df.shift(1, ['x'])
    dfn1 = df.shift(-1, ['x'])
    i2 = i1 + length + 1
    assert dfp1[i1:i2].x.tolist() == [None, 0, 1, 2, None][i1:i2]
    assert dfp1[i1:i2].y.tolist() == [0, 1, None, 9, 16][i1:i2]
    assert dfn1[i1:i2].x.tolist() == [1, 2, None, 4, None][i1:i2]

def test_shift_basics_trim(df_factory):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    dfp1 = df.shift(1, ['x'], trim=True)
    dfn1 = df.shift(-1, ['x'], trim=True)
    assert dfp1.x.tolist() == [0, 1, 2, None]
    assert dfp1.y.tolist() == [1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4]
    assert dfn1.y.tolist() == [0, 1, None, 9]

    assert dfp1.shift(1, trim=True).x.tolist() == [0, 1, 2]
    assert dfp1.shift(-1, trim=True).x.tolist() == [1, 2, None]


def test_shift_range(df_factory):
    x = [0, 1, 2, 3, 4]
    xm1 = [1, 2, 3, 4, None]
    y = [0, 1, None, 9, 16]
    df = df_factory(x=x, y=y)
    df['x1'] = df['x']
    df['x2'] = df['x']
    df.shift(0, ['x1'], inplace=True)
    df.shift(-1, ['x2'], inplace=True)
    assert df.x1.tolist() == x
    assert df.x2.tolist() == xm1
    assert df.func.stack([df.x1, df.x2]).tolist() == [[0, 1], [1, 2], [2, 3], [3, 4], [4, None]]
    df = df_factory(x=x, y=y)
    df.shift((0, 2), 'x', inplace=True)
    assert df.x.tolist() == [[0, 1], [1, 2], [2, 3], [3, 4], [4, None]]

    # trim with range
    df = df_factory(x=x, y=y)
    df.shift((0, 3), 'x', inplace=True, trim=True)
    assert df.x.tolist() == [[0, 1, 2], [1, 2, 3], [2, 3, 4]]



def test_shift_filtered(df_factory):
    x = [0, 99, 1, 99, 2, 99, None, 99, 4, 99]
    y = [0, 88, 1, 88, None, 88, 9, 88, 16, 88]
    assert len(x) == len(y)
    df = df0 = df_factory(x=x, y=y)
    df = df[((df.x != 99) | df.x.ismissing()).fillna(True)]
    dfp1 = df.shift(1, ['x'])
    dfn1 = df.shift(-1, ['x'])
    assert dfp1.x.tolist() == [None, 0, 1, 2, None]
    assert dfp1.y.tolist() == [0, 1, None, 9, 16]
    assert dfn1.x.tolist() == [1, 2, None, 4, None]
    assert dfn1.y.tolist() == [0, 1, None, 9, 16]

    assert dfp1.shift(1).x.tolist() == [None, None, 0, 1, 2]
    assert dfp1.shift(-1).x.tolist() == [0, 1, 2, None, None]
    assert dfp1.shift(-1, fill_value=99).x.tolist() == [0, 1, 2, None, 99]

    assert dfn1.shift(1).x.tolist() == [None, 1, 2, None, 4]
    assert dfn1.shift(-1).x.tolist() == [2, None, 4, None, None]
    assert dfn1.shift(-1, fill_value=99).x.tolist() == [2, None, 4, None, 99]

    assert df.shift(4).x.tolist() == [None, None, None, None, 0]
    assert df.shift(5).x.tolist() == [None, None, None, None, None]
    assert df.shift(6).x.tolist() == [None, None, None, None, None]

    assert df.shift(-4).x.tolist() == [4, None, None, None, None]
    assert df.shift(-5).x.tolist() == [None, None, None, None, None]
    assert df.shift(-6).x.tolist() == [None, None, None, None, None]


def test_shift_string(df_factory):
    x = np.arange(4)
    s = pa.array(['aap', None, 'noot', 'mies'])
    df = df_factory(x=x, s=s)
    assert df.shift(1).s.tolist() == [None, 'aap', None, 'noot']
    assert df.shift(-1).s.tolist() == [None, 'noot', 'mies', None]
    assert df.shift(1, ['s'], fill_value='VAEX').s.tolist() == ['VAEX', 'aap', None, 'noot']
    assert df.shift(-1, ['s'], fill_value='VAEX').s.tolist() == [None, 'noot', 'mies', 'VAEX']


def test_shift_virtual(df_factory):
    x = [0, 1, 2, None, 4]
    y = [0, 1, None, 9, 16]
    xsp1 = [None, 0, 1, 2, None]
    xsn1 = [1, 2, None, 4, None]
    df = df_factory(x=x, y=y)

    # # a is a virtual column that depends on x, but we don't shift a
    df['a'] = df.x + 0
    df['b'] = df.a
    dfs = df.shift(1, ['x'])
    assert dfs.x.tolist() == xsp1
    assert dfs.a.tolist() == x
    assert dfs.y.tolist() == y
    dfs = df.shift(-1, ['x'])
    assert dfs.x.tolist() == xsn1
    assert dfs.a.tolist() == x
    assert dfs.y.tolist() == y

    # a is a virtual column that depends on x, we shift a, but we don't shift x
    # we expect, a: __x_shifted, x: __x
    df = df_factory(x=x, y=y)
    df['a'] = df.x + 0
    dfs = df.shift(1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsp1
    assert dfs.y.tolist() == y
    dfs = df.shift(-1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsn1
    assert dfs.y.tolist() == y

    # same, but now we also have a reference to a, which we also do not shift
    df = df_factory(x=x, y=y)
    df['a'] = df.x + 0
    df['b'] = df.a + 0
    dfs = df.shift(1, ['a'])
    assert dfs.x.tolist() == x
    assert dfs.a.tolist() == xsp1
    assert dfs.b.tolist() == x
    assert dfs.y.tolist() == y
    dfs = df.shift(-1, ['a'])
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

@pytest.mark.parametrize("chunk_number", [0.5, 1, 2.5, 5.5])
@pytest.mark.parametrize("period", list(range(-3, 4)))
def test_shift_large_dataset(chunk_number, period):
    chunk_size = 1024**2 # same value at _chunk_iterator()

    v=np.random.random(int(chunk_number*chunk_size))

    df = vaex.from_arrays(x=v)

    w = df.shift(period).values.reshape(-1)

    if period<0:
        assert np.all(w[:period]==v[-period:])
        assert w[period:].tolist() == [None]*(-period)
    elif period>0:
        assert np.all(w[period:]==v[:-period])
        assert w[:period].tolist() == [None]*period
    else:
        assert np.all(w==v)

@pytest.mark.parametrize("periods", [-1, 1, 2, -2])
def test_diff(df_factory, periods):
    x = [0, 1, 2, 3, 4.0]
    df = df_factory(x=x)
    dfp = df.to_pandas_df(array_type='numpy')
    df = df.diff(periods, fill_value=np.nan)
    dfp = dfp.diff(periods)
    result = df['x'].to_numpy()
    expected = dfp['x'].to_numpy()
    assert np.all(np.isnan(result) == np.isnan(expected))
    mask = ~np.isnan(result)
    assert result[mask].tolist() == expected[mask].tolist()

def test_diff_list():
    periods = 2
    x = np.arange(10, dtype='f8')
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    dfp = df.to_pandas_df(array_type='numpy')
    df = df.diff(periods, fill_value=np.nan, column=['x', 'y'])
    dfp = dfp.diff(periods)
    result = df['x'].to_numpy()
    expected = dfp['x'].to_numpy()
    assert np.all(np.isnan(result) == np.isnan(expected))
    mask = ~np.isnan(result)
    assert result[mask].tolist() == expected[mask].tolist()
    
@pytest.mark.parametrize("chunk_number", [0.5, 1, 2.5, 5.5])
def test_diff_large_dataset(chunk_number):
    chunk_size = 1024**2 # same value at _chunk_iterator()

    v=np.random.random(int(chunk_number*chunk_size))

    df = vaex.from_arrays(x=v)

    w = df.diff().values.reshape(-1)

    assert np.all(w[1:]==np.diff(v))
    assert w[:1].tolist()==[None]

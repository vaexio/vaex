import pytest
import vaex
import vaex.schema
import numpy as np
import pyarrow as pa


def test_concat_endianness(df_concat, df_trimmed):
    assert df_trimmed.data_type('x').byteorder == '>'
    # since df_concat will concate, we will lose the byteorder
    assert df_concat.data_type('x').byteorder in '<='


def test_concat():
    x1, y1, z1 = np.arange(3), np.arange(3, 0, -1), np.arange(10, 13)
    x2, y2, z2 = np.arange(3, 6), np.arange(0, -3, -1), np.arange(13, 16)
    x3, y3, z3 = np.arange(6, 9), np.arange(-3, -6, -1), np.arange(16, 19)
    w1, w2, w3 = np.array(['cat']*3), np.array(['dog']*3), np.array(['fish']*3)
    x = np.concatenate((x1, x2, x3))
    y = np.concatenate((y1, y2, y3))
    z = np.concatenate((z1, z2, z3))
    w = np.concatenate((w1, w2, w3))

    ds = vaex.from_arrays(x=x, y=y, z=z, w=w)
    ds1 = vaex.from_arrays(x=x1, y=y1, z=z1, w=w1)
    ds2 = vaex.from_arrays(x=x2, y=y2, z=z2, w=w2)
    ds3 = vaex.from_arrays(x=x3, y=y3, z=z3, w=w3)

    dd = vaex.concat([ds1, ds2])
    ww = ds1.concat(ds2)

    # Test if the concatination of two arrays with the vaex method is the same as with the dataset method
    assert (np.array(dd.evaluate('x,y,z,w'.split(','))) == np.array(ww.evaluate('x,y,z,w'.split(',')))).all()

    # Test if the concatination of multiple datasets works
    dd = vaex.concat([ds1, ds2, ds3])
    assert (np.array(dd.evaluate('x')) == np.array(ds.evaluate('x'))).all()
    assert (np.array(dd.evaluate('y')) == np.array(ds.evaluate('y'))).all()
    assert (np.array(dd.evaluate('z')) == np.array(ds.evaluate('z'))).all()
    assert (np.array(dd.evaluate('w')) == np.array(ds.evaluate('w'))).all()

    # Test if the concatination of concatinated datasets works
    dd1 = vaex.concat([ds1, ds2])
    dd2 = vaex.concat([dd1, ds3])
    assert (np.array(dd2.evaluate('x')) == np.array(ds.evaluate('x'))).all()
    assert (np.array(dd2.evaluate('y')) == np.array(ds.evaluate('y'))).all()
    assert (np.array(dd2.evaluate('z')) == np.array(ds.evaluate('z'))).all()
    assert (np.array(dd2.evaluate('w')) == np.array(ds.evaluate('w'))).all()


def test_concat_unequals_virtual_columns():
    ds1 = vaex.from_scalars(x=1, y=2)
    ds2 = vaex.from_scalars(x=2, y=3)
    # w has same expression
    ds1['w'] = ds1.x + ds1.y
    ds2['w'] = ds2.x + ds2.y
    # z does not
    ds1['z'] = ds1.x + ds1.y
    ds2['z'] = ds2.x * ds2.y
    ds = vaex.concat([ds1, ds2])
    assert ds.w.tolist() == [1+2, 2+3]
    assert ds.z.tolist() == [1+2, 2*3]


def test_concat_keep_virtual():
    df1 = vaex.from_scalars(x=1, y=2)
    df2 = vaex.from_scalars(x=2, y=3)
    # w has same expression
    df1['w'] = df1.x + df1.y
    df2['w'] = df2.x + df2.y
    df = vaex.concat([df1, df2])
    assert 'w' in df.virtual_columns
    assert 'w' not in df.get_column_names(virtual=False)
    assert 'w' not in df.dataset


def test_concat_dtype_object(df_concat):
    df = df_concat
    assert df.obj.tolist()


def test_concat_arrow_strings():
    df1 = vaex.from_arrays(x=vaex.string_column(['aap', 'noot', 'mies']))
    df2 = vaex.from_arrays(x=vaex.string_column(['a', 'b', 'c']))
    df = vaex.concat([df1, df2])
    assert df.data_type('x') == df1.data_type('x')
    assert df.x.tolist() == ['aap', 'noot', 'mies', 'a', 'b', 'c']


def test_concat_mixed_types():
    x1 = np.zeros(3) + np.nan
    x2 = vaex.string_column(['hi', 'there'])
    df1 = vaex.from_arrays(x=x1)
    df2 = vaex.from_arrays(x=x2)
    df = vaex.concat([df1, df2])
    assert df2.x.dtype == df.x.dtype, "expect 'upcast' to string"
    assert df[:2].x.tolist() == ['nan', 'nan']
    assert df[1:4].x.tolist() == ['nan', 'nan', 'hi']
    assert df[2:4].x.tolist() == ['nan', 'hi']
    assert df[3:4].x.tolist() == ['hi']
    assert df[3:5].x.tolist() == ['hi', 'there']


def test_dtypes(df_concat):
    assert df_concat.timedelta.dtype.kind == 'm'
    assert df_concat.datetime.dtype.kind == 'M'


def test_hidden(df_trimmed):
    xlist = (df_trimmed.x + 1).tolist()
    df = df_trimmed.copy()
    df['x'] = df.x + 1
    dfc = df.concat(df)  # make sure concat copies hidden columns
    assert dfc.x.tolist() == xlist + xlist


def test_concat_filtered(df_trimmed):
    df = df_trimmed
    df1 = df[df.x < 5]
    df2 = df[df.x >= 5]
    dfc = df1.concat(df2)
    assert dfc.x.tolist() == df.x.tolist()


@pytest.mark.parametrize("i1", list(range(0, 6)))
@pytest.mark.parametrize("length", list(range(1, 6)))
def test_sliced_concat(i1, length, df_concat):
    i2 = i1 + length
    x = df_concat.x.tolist()
    df = df_concat[i1:i2]
    assert len(df_concat[:length]['x'].values) == length
    assert df.x.tolist() == x[i1:i2]


def test_concat_masked_values(df_concat):
    df = df_concat
    # evaluate a piece not containing masked values
    assert df.m.tolist(0, 3) == df.x[:3].tolist()


def test_concat_masked_with_unmasked():
    # Exposes https://github.com/vaexio/vaex/issues/661
    df1 = vaex.from_arrays(x=np.arange(3.))
    df2 = vaex.from_arrays(x=np.ma.masked_all(2, dtype='f8'))
    df = df1.concat(df2)
    assert df.x.tolist() == [0, 1, 2, None, None]
    assert df.is_masked('x')


def test_concat_virtual_column_names():
    df1 = vaex.from_arrays(x=np.arange(3.))
    df2 = vaex.from_arrays(x=np.arange(4.))
    df1['z'] = df1.x ** 2
    df2['z'] = df2.x ** 2
    df = df1.concat(df2)
    # test that we get the correct column names (we had duplicates before)
    assert df.get_column_names() == ['x', 'z']
    assert list(df) == ['x', 'z']


def test_concat_missing_values():
    df1 = vaex.from_arrays(x=[1, 2, 3], y=[np.nan, 'b', 'c'])
    df2 = vaex.from_arrays(x=[4, 5, np.nan], y=['d', 'e', 'f'])
    df = vaex.concat([df1, df2])

    repr(df.head(4))
    repr(df.tail(4))
    assert len(df) == 6


def test_schema_flexible():
    assert vaex.schema.resolver_flexible.resolve([np.dtype('f8'), np.dtype('f8')], [()]) == (np.dtype('f8'), ())
    assert vaex.schema.resolver_flexible.resolve([np.dtype('f8'), np.dtype('f4')], [()]) == (np.dtype('f8'), ())
    assert vaex.schema.resolver_flexible.resolve([np.dtype('f8'), np.dtype('i4')], [()]) == (np.dtype('f8'), ())
    assert vaex.schema.resolver_flexible.resolve([np.dtype('i4'), np.dtype('i2')], [()]) == (np.dtype('i4'), ())
    # eagerly upcast to arrow
    assert vaex.schema.resolver_flexible.resolve([pa.float64(), np.dtype('i2')], [()]) == (pa.float64(), ())


def test_mixed_type_numpy():
    def concat(*types):
        dfs = [vaex.from_arrays(x=np.arange(3, dtype=dtype)) for dtype in types]
        dataset_concat = vaex.concat(dfs)
        return dataset_concat

    assert concat(np.float32, np.float64).columns["x"].dtype == np.float64
    assert concat(np.float32, np.int64).columns["x"].dtype == np.float64
    assert concat(np.float32, np.byte).columns["x"].dtype == np.float32
    assert concat(np.float64, np.byte, np.int64).columns["x"].dtype == np.float64


@pytest.mark.parametrize("arrow", [True, False])
def test_concat_unaligned_schema(arrow):
    x = pa.array([1, 2], type=pa.float32()) if arrow else np.array([1, 2], dtype=np.float32())
    df1 = vaex.from_arrays(x=x)
    df2 = vaex.from_arrays(y=['d', 'e'])
    # flexible should be the default
    df = df1.concat(df2, resolver='flexible')
    df = df1.concat(df2)
    assert df.x.tolist() == [1, 2, None, None]
    assert df.y.tolist() == [None, None, 'd', 'e']
    # always 'upcast' to Arrow arrays
    # # rationale: Arrow will use use less memory, numpy has no efficient way to represent all missing data
    assert df.x.data_type() == pa.float32()

    # strict resolver to now allows mismatched schemas (pre v4.0 behaviour)
    with pytest.raises(NameError):
        df = df1.concat(df2, resolver='strict')


@pytest.mark.parametrize("typed", [True, False])
def test_concat_with_null_type(typed):
    # when concatting an Arrow float32 column with a null type, resulting type should be Arrow float32
    arrow_type = pa.float32()
    x1 = pa.array([1, 2], type=arrow_type)
    x2 = pa.nulls(2, type=arrow_type if typed else None)
    df1 = vaex.from_arrays(x=x1)
    df2 = vaex.from_arrays(x=x2)
    df = df1.concat(df2)
    assert df.x.data_type() == arrow_type
    assert df.x.tolist() == [1, 2, None, None]


@pytest.mark.parametrize("typed", [True, False])
def test_concat_with_null_type_numpy_and_arrow(typed):
    # when concatting a numpy float32 column with a null type, resulting type should be Arrow float32
    # rationale: Arrow will use use less memory, numpy has no efficient way to represent all missing data
    numpy_type = np.float32()
    arrow_type = pa.float32()
    x1 = np.array([1, 2], dtype=numpy_type)
    x2 = pa.nulls(2, type=arrow_type if typed else None)
    df1 = vaex.from_arrays(x=x1)
    df2 = vaex.from_arrays(x=x2)
    df = df1.concat(df2)
    assert df.x.data_type() == arrow_type
    assert df.x.tolist() == [1, 2, None, None]

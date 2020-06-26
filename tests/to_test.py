import xarray
import pyarrow as pa


def test_to_items(df_local):
    df = df_local
    items = df.to_items(['x', 'y'])
    (xname, xvalues), (yname, yvalues) = items
    assert xname == 'x'
    assert yname == 'y'
    assert xvalues.tolist() == df.x.tolist()
    assert yvalues.tolist() == df.y.tolist()

    for i1, i2, items in df.to_items(['x', 'y'], chunk_size=3):
        (xcname, xcvalues), (ycname, ycvalues) = items
        assert xcname == 'x'
        assert ycname == 'y'
        assert xcvalues.tolist() == xvalues[i1:i2].tolist()
        assert ycvalues.tolist() == yvalues[i1:i2].tolist()


def test_to_array_type_list(df_local):
    df = df_local
    data = df[:3]['x', 'y'].to_dict(array_type='list')
    assert data == {'x': [0, 1, 2], 'y': [0, 1, 4]}
    assert isinstance(data['x'], list)


def test_to_array_type_xarray(df_local):
    df = df_local
    data = df[:3]['x', 'y'].to_dict(array_type='xarray')
    assert isinstance(data['x'], xarray.DataArray)
    assert data['x'].data.tolist() == [0, 1, 2]
    assert data['y'].data.tolist() == [0, 1, 4]


def test_to_arrow_table(df_local):
    df = df_local
    t = df.to_arrow_table(['x', 'y'])
    record_batches = t.to_batches(3)
    index = 0
    for i1, i2, tc in df.to_arrow_table(['x', 'y'], chunk_size=3):
        record_batches[index].to_pydict() == tc.to_pydict()
        index += 1


def test_to_pandas_df(df_local):
    df = df_local
    pdf = df.to_pandas_df(['x', 'y'], index_name='x')
    assert pdf.index.name == 'x'
    assert pdf.columns == ['y']
    assert pdf.index.values.tolist() == df.x.tolist()
    assert pdf.y.values.tolist() == df.y.tolist()
    x = df.x.values
    y = df.y.values

    for i1, i2, pdf in df.to_pandas_df(['y'], index_name='x', chunk_size=3):
        assert pdf.index.name == 'x'
        assert pdf.columns == ['y']
        assert pdf.index.values.tolist() == x[i1:i2].tolist()
        assert pdf.y.values.tolist() == y[i1:i2].tolist()


def test_to_arrow_arrays(df_local):
    df = df_local
    assert isinstance(df['x'].to_arrow(convert_to_native=True), pa.Array)
    # assert isinstance(pa.array(df['x']), pa.Array)

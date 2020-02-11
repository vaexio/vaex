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

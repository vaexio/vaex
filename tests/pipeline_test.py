

def test_virtual_column(df_factory_numpy, tmpdir):
    df = df_factory_numpy(x=[1, 2])
    dfc = df.copy()
    df['y'] = df.x + 1
    dft = df.pipeline.transform(dfc)
    assert dft['y'].tolist() == df['y'].tolist()

    file = str(tmpdir / 'pipeline.json')
    df.pipeline.save(file)
    dft = dfc.pipeline.load_and_transform(file)
    assert dft['y'].tolist() == df['y'].tolist()



    # def add(df, col, value):
    #     df[col] = df[col] + value
    #     return df
    # df2 = df.transform(add, 'y', 2)
    # assert df2['y'].tolist() == [3, 4, 5, 6, 7, 8]


def test_ml(df_factory_numpy, tmpdir):
    df = df_factory_numpy(x=[1, 2, 3])
    dfc = df.copy()
    df = df.ml.minmax_scaler(features=['x'])
    assert df.minmax_scaled_x.tolist() == [0, 0.5, 1.]
    dft = df.pipeline.transform(dfc)
    assert dft['minmax_scaled_x'].tolist() == df['minmax_scaled_x'].tolist()

    file = str(tmpdir / 'pipeline.json')
    df.pipeline.save(file)
    dft = dfc.pipeline.load_and_transform(file)
    assert dft['x'].tolist() == df['x'].tolist()
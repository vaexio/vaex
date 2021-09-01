from common import *

def test_materialize(df_factory_arrow):
    df = df_factory_arrow(x=[1, 2], y=[3, 5])
    dfc = df.copy()

    df['r'] = np.sqrt(df.x**2 + df.y**2)
    assert 'r' in df.virtual_columns
    assert hasattr(df, 'r')
    df = df.materialize(df.r)
    assert 'r' not in df.virtual_columns
    assert 'r' in df.columns
    assert hasattr(df, 'r')
    assert df.r.evaluate().tolist() == np.sqrt(df.x.to_numpy()**2 + df.y.to_numpy()**2).tolist()

    df2 = df.pipeline.transform(dfc)
    assert 'r' not in df2.virtual_columns
    assert 'r' in df2.columns
    assert hasattr(df2, 'r')
    assert df2.r.evaluate().tolist() == np.sqrt(df.x.to_numpy()**2 + df.y.to_numpy()**2).tolist()

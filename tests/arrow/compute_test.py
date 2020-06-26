
def test_mul(df_trimmed):
    df = df_trimmed
    x = df.x.values
    y = df.y.values
    df['x'] = df.x.as_arrow()
    df['y'] = df.y.as_arrow()
    df['p'] = df.x.as_numpy() * df.y.as_numpy()
    assert df.p.tolist() == (x * y).tolist()

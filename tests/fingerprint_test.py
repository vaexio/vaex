
def test_dataframe(df_factory):
    df1 = df_factory(x=[1, 2], y=[4, 5])
    df1b = df_factory(x=[1, 2], y=[4, 5])
    df2 = df_factory(x=[1, 3], y=[4, 5])

    assert df1.fingerprint() == df1b.fingerprint()
    assert df1.fingerprint() != df2.fingerprint()

    assert df1.fingerprint() == df1b.fingerprint()
    df1.add_variable('q', 1)  # this changes the state
    assert df1.fingerprint() != df1b.fingerprint()


def test_groupby(df_factory):
    df1 = df_factory(x=[1, 2], y=[4, 5])
    df2 = df_factory(x=[1, 2], y=[4, 5])

    df1g = df1.groupby('x', agg='count', sort=True)
    df2g = df2.groupby('x', agg='count', sort=True)

    assert df1g.fingerprint() == df2g.fingerprint()


def test_expression(df_factory):
    df1 = df_factory(x=[1, 2], y=[4, 5])
    df1b = df_factory(x=[1, 2], y=[4, 5])
    df2 = df_factory(x=[1, 3], y=[4, 5])
    df1['z'] = 'x + y'
    df1b['z'] = 'x + y'
    df2['z'] = 'x + y'
    assert df1.x.fingerprint() == df1b.x.fingerprint()
    assert df1.y.fingerprint() == df1b.y.fingerprint()
    assert df1.z.fingerprint() == df1b.z.fingerprint()

    assert df1.z.fingerprint() != df2.z.fingerprint()
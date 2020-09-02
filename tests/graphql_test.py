from common import *


if vaex.utils.devmode:
    pytest.skip('runs too slow when developing', allow_module_level=True)

@pytest.fixture(scope='module')
def schema(ds_trimmed_cache):
    ds_trimmed_cache = ds_trimmed_cache.drop('123456')
    return ds_trimmed_cache.graphql.schema()

@pytest.fixture()
def df(df_trimmed):
    return df_trimmed.drop('123456')


def test_aggregates(df, schema):
    result = schema.execute("""
    {
        df {
            count
            min {
                x
                y
            }
            mean {
                x
                y
            }
            max {
                x
                y
            }
        }
    }
    """)
    assert not result.errors
    assert result.data['df']['count'] == len(df)
    assert result.data['df']['min']['x'] == df.x.min()
    assert result.data['df']['min']['y'] == df.y.min()
    assert result.data['df']['max']['x'] == df.x.max()
    assert result.data['df']['max']['y'] == df.y.max()
    assert result.data['df']['mean']['x'] == df.x.mean()
    assert result.data['df']['mean']['y'] == df.y.mean()


def test_groupby(df, schema):

    result = schema.execute("""
    {
        df {
            groupby {
                x {
                    min {
                        x
                    }
                }
            }
        }
    }
    """)
    assert not result.errors
    dfg = df.groupby('x', agg={'xmin': vaex.agg.min('x')})
    assert result.data['df']['groupby']['x']['min']['x'] == dfg['xmin'].tolist()


def test_row_pagination(df, schema):
    def values(row, name):
        return [k[name] for k in row]
    result = schema.execute("""
    {
        df {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df.x.tolist()

    result = schema.execute("""
    {
        df {
            row(offset: 2) { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[2:].x.tolist()

    result = schema.execute("""
    {
        df {
            row(limit: 2) { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[:2].x.tolist()

    result = schema.execute("""
    {
        df {
            row(offset: 3, limit: 2) { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[3:5].x.tolist()


def test_where(df, schema):
    def values(row, name):
        return [k[name] for k in row]
    result = schema.execute("""
    {
        df(where: {x: {_eq: 4}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x==4].x.tolist()

    result = schema.execute("""
    {
        df(where: {x: {_neq: 4}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x!=4].x.tolist()

    result = schema.execute("""
    {
        df(where: {x: {_gt: 4}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x>4].x.tolist()

    result = schema.execute("""
    {
        df(where: {x: {_gte: 4}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x>=4].x.tolist()

    result = schema.execute("""
    {
        df(where: {x: {_lt: 4}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x<4].x.tolist()

    result = schema.execute("""
    {
        df(where: {x: {_lte: 4}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[df.x<=4].x.tolist()

    result = schema.execute("""
    {
        df(where: {_not: {x: {_lte: 4}}}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == df[~(df.x<=4)].x.tolist()

    result = schema.execute("""
    {
        df(where: {_or: [{x: {_eq: 4}}, {x: {_eq: 6}} ]}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == [4, 6]

    result = schema.execute("""
    {
        df(where: {_and: [{x: {_gte: 4}}, {x: {_lte: 6}} ]}) {
            row { x }
        }
    }
    """)
    assert not result.errors
    assert values(result.data['df']['row'], 'x') == [4, 5, 6]


def test_pandas(df, schema):
    df_pandas = df.to_pandas_df()
    def values(row, name):
        return [k[name] for k in row]
    result = df_pandas.graphql.execute("""
    {
        df(where: {x: {_eq: 4}}) {
            row { x }
        }
    }
    """)
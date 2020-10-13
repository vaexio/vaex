import pytest
import vaex
import pyarrow as pa


@pytest.mark.parametrize('op', vaex.expression._binary_ops)
def test_binary_ops(df_trimmed, op):
    if op['name'] in ['contains', 'and', 'xor', 'or', 'is', 'is_not', 'matmul']:
        return

    operator = op['op']
    df = df_trimmed
    x = df.x.astype(pa.int32()).to_numpy()
    y = df.y.astype(pa.int64()).to_numpy()
    df['x'] = df.x.astype(pa.int32()).as_arrow()
    df['y'] = df.y.astype(pa.int64()).as_arrow()
    df['pa'] = operator(df.x, 1 + df.y)
    assert df.pa.tolist() == operator(x, 1 + y).tolist()
    assert df.pa.values.to_pylist() == operator(x, 1 + y).tolist()


@pytest.mark.parametrize("as_numpy", [False, True])
def test_selections_using_arrow_strings(as_numpy):
    schema = pa.schema({'x': pa.float64(),
                        's': pa.string()
                        })
    data = {'x': [1, 1, 2, 2, 3, 3, 5],
            's': ['blue', 'blue', 'red', 'red', None, 'green', 'yellow']
            }

    arrow_data = pa.Table.from_pydict(mapping=data, schema=schema)
    df = vaex.from_arrow_table(arrow_data, as_numpy=as_numpy)

    s_count = df.count(selection='s == "blue"')
    assert s_count == 2

    x_count = df.count(selection='x == 2')
    assert x_count == 2

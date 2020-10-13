import vaex
import pyarrow as pa
import pytest


@pytest.mark.parametrize("as_numpy", [False, True])
def test_where_string_dtype(as_numpy):
    schema = pa.schema({'x': pa.float64(),
                        's': pa.string(),
                        })
    data = {'x': [1, 1, 2, 2, 3, 3, 5],
            's': ['blue', 'blue', 'red', 'red', None, 'green', 'yellow']
            }

    arrow_data = pa.Table.from_pydict(mapping=data, schema=schema)
    df = vaex.from_arrow_table(arrow_data, as_numpy=as_numpy)

    df['s_modified'] = df.func.where(df.s == 'red', 'pink', df.s)

    assert df.s_modified.is_string()

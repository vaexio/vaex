import pytest

import pandas as pd
import pyarrow as pa
import vaex
import pytest

df = pd.DataFrame({'col1': range(5)})
table = pa.Table.from_pandas(df)


@pytest.mark.parametrize("as_stream", [True, False])
def test_arrow_write_table(tmpdir, as_stream):
    path = str(tmpdir.join('test.arrow'))
    vaex.from_arrow_table(table).export_arrow(path, as_stream=as_stream)
    df = vaex.open(path)
    assert 'col1' in df


def test_chunks(df_trimmed, tmpdir):
    path = str(tmpdir.join('test.arrow'))
    df = df_trimmed[['x', 'y', 'name']]
    df.export_arrow(path, chunk_size=2)
    df_read = vaex.open(path)
    assert isinstance(df_read.columns['x'], pa.ChunkedArray)
    assert df_read.x.tolist() == df.x.tolist()


@pytest.mark.parametrize("as_numpy", [False, True])
def test_from_arrow_table(as_numpy):
    schema = pa.schema({'x': pa.float64(),
                        's': pa.string(),
                        })

    data = {'x': [1, 1, 2, 2, 3, 3, 5],
            's': ['blue', 'blue', 'red', 'red', None, 'green', 'yellow']
            }

    arrow_data = pa.Table.from_pydict(mapping=data, schema=schema)
    df = vaex.from_arrow_table(arrow_data, as_numpy=as_numpy)

    assert df.shape == (7, 2)
    assert df.x.dtype == 'float64'
    assert df.s.dtype == 'str'  # while df.is_string(df.s) returns True always, but df.x.dtype returns 'O' for as_numpy=True

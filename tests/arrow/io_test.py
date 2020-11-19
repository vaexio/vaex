import pytest

import pandas as pd
import pyarrow as pa
import vaex

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

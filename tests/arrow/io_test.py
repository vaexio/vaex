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


@pytest.mark.parametrize("as_stream", [True, False])
def test_empty(tmpdir, as_stream):
    path = str(tmpdir.join('test.arrow'))
    schema = pa.schema([pa.field("x", pa.string())])
    with open(path, mode='wb') as f:
        if as_stream:
            writer = pa.ipc.new_stream(f, schema)
        else:
            writer = pa.ipc.new_file(f, schema)
        writer.close()
    vaex.open(path)


@pytest.mark.parametrize("filename", ['test.parquet', 'test.arrow'])
def test_empty(tmpdir, filename):
    df = vaex.from_arrays(x=[1,2])
    path = tmpdir / filename
    dff = df[df.x > 3]
    dff.export(path)
    df2 = vaex.open(path)
    assert df2.x.tolist() == []

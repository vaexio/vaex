import pandas as pd
import pyarrow as pa
import vaex

df = pd.DataFrame({'col1': range(5)})
table = pa.Table.from_pandas(df)


def test_arrow_write_table(tmpdir):
    path = str(tmpdir.join('test.arrow'))
    with pa.OSFile(path, 'wb') as sink:
        with pa.RecordBatchFileWriter(sink, table.schema) as writer:
            writer.write_table(table)

    df = vaex.open(path)


def test_arrow_write_stream(tmpdir):
    path = str(tmpdir.join('test.arrow'))
    with pa.OSFile(path, 'wb') as sink:
        with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
            writer.write_table(table)

    df = vaex.open(path)


def test_chunks(df_trimmed, tmpdir):
    path = str(tmpdir.join('test.arrow'))
    df = df_trimmed[['x', 'y', 'name']]
    df.export_arrow_stream(path, chunk_size=2)
    df_read = vaex.open(path, as_numpy=False)
    assert isinstance(df_read.columns['x'], pa.ChunkedArray)
    assert df_read.x.tolist() == df.x.tolist()

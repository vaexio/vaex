import pyarrow as pa
import numpy as np
import vaex
import pandas as pd
import pytest


pdf = pd.DataFrame(
    {
        'col1': pd.Categorical.from_codes(np.full(1, 1), categories=['ABC', 'DEF'])
    }
)

@pytest.mark.xfail(reason="not commited, ")
def test_categorical(tmpdir):
    # based on https://github.com/vaexio/vaex/issues/399
    path = str(tmpdir.join('test.arrow'))
    table = pa.Table.from_pandas(pdf)

    with pa.OSFile(path, 'wb') as sink:
        with pa.RecordBatchStreamWriter(sink, table.schema) as writer:
            writer.write_table(table)
    with pa.OSFile(path, 'rb') as source:
        pdf2 = pa.ipc.open_stream(source).read_pandas()


    df = vaex.open(path)
    assert df.col1.tolist() == ["DEF"]
    assert df.is_category(df.col1)
    assert df.category_labels(df.col1) == ['ABC', 'DEF']

import pyarrow as pa
import pyarrow.parquet as pq
import vaex
import numpy as np
import os
from pathlib import Path

path = Path(__file__).parent.parent
data_path = path / 'data'
parquet_path = os.path.join(path, 'data', 'sample_arrow_dict.parquet')


def test_dict_col(tmpdir):
    # Create the file if necessary
    parquet_path = tmpdir / 'sample_arrow_dict.parquet'
    schema = pa.schema({
        'col1': pa.int32(),
        'col2': pa.float32(),
        'col3': pa.dictionary(pa.int16(), pa.string()),
    })

    table = pa.table({
        'col1': range(10),
        'col2': np.random.randn(10),
        'col3': list(np.random.choice(['A', 'B', 'C'], 10)),
    }, schema=schema)

    pq.write_table(table, parquet_path)

    # Load df
    df = vaex.open(parquet_path)
    dtypes = df.dtypes
    assert isinstance(dtypes["col3"].arrow, pa.lib.DictionaryType)

    # Filter
    df = df._future()
    dff1 = df[df["col3"] == 'A']
    assert dff1["col3"].unique() == ["A"]

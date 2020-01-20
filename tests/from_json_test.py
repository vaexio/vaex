from common import *
import tempfile


def test_from_json(ds_local):
    df = ds_local

    # Create temporary json files
    pandas_df = df.to_pandas_df(df, virtual=True)
    tmp = tempfile.mktemp('.json')
    with open(tmp, 'w') as f:
        f.write(pandas_df.to_json())

    tmp_df = vaex.from_json(tmp, copy_index=False)

    assert set(tmp_df.column_names) == set(df.column_names)
    assert len(tmp_df) == len(df)
    assert tmp_df.x.tolist() == df.x.tolist()
    assert tmp_df.bool.tolist() == df.bool.tolist()

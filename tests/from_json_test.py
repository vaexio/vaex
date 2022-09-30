from common import *
import tempfile

import vaex


def test_from_json(ds_local):
    df = ds_local

    # Create temporary json files
    pandas_df = df.to_pandas_df(virtual=True, array_type='numpy')
    tmp = tempfile.mktemp('.json')
    with open(tmp, 'w') as f:
        f.write(pandas_df.to_json())

    tmp_df = vaex.from_json(tmp)

    assert set(tmp_df.get_column_names()) == set(df.get_column_names())
    assert len(tmp_df) == len(df)
    assert tmp_df.x.tolist() == df.x.tolist()
    assert tmp_df.bool.tolist() == df.bool.tolist()


@pytest.mark.parametrize("backend", ["pandas", "json"])
@pytest.mark.parametrize("lines", [False, True])
def test_from_and_export_json(tmpdir, ds_local, backend, lines):
    df = ds_local
    df = df.drop(columns=['datetime'])
    if 'timedelta' in df:
        df = df.drop(columns=['timedelta'])
    if 'obj' in df:
        df = df.drop(columns=['obj'])

    # Create temporary json files
    tmp = str(tmpdir.join('test.json'))
    df.export_json(tmp, backend=backend, lines=lines)

    # Check if file can be read with default (pandas) backend
    df_read = vaex.from_json(tmp, lines=lines)
    assert df.shape == df_read.shape
    assert df.x.tolist() == df_read.x.tolist()
    assert df.get_column_names() == df_read.get_column_names()

    # If lines is True, check if the file can be read with the from_json_arrow function
    if lines:
        df_read_arrow = vaex.from_json_arrow(tmp)
        assert df.shape == df_read_arrow.shape
        assert df.x.tolist() == df_read_arrow.x.tolist()
        assert df.get_column_names() == df_read_arrow.get_column_names()

from common import ds_local


def test_nop(ds_local):
    df = ds_local
    column_names = df.column_names

    # Try a nop on a single column
    result = df.nop(column_names[1])
    assert result is None

    # Try a  nop on a list of columns
    result = df.nop(column_names)
    assert result is None

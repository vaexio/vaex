from vaex.ml.datasets import load_titanic


def test_get_column_names_dtypes():
    df = load_titanic()
    assert len(df.get_column_names(dtypes=int)) == 3
    assert len(df.get_column_names(dtypes=[int])) == 3
    assert len(df.get_column_names(dtypes=[float])) == 3
    assert len(df.get_column_names(dtypes=[str])) == 7
    assert len(df.get_column_names(dtypes=[object])) == 7
    assert len(df.get_column_names(dtypes=[bool])) == 1
    assert len(df.get_column_names(dtypes=[str, int])) == 10
    assert len(df.get_column_names(dtypes=[str, int, float])) == 13
    assert len(df.get_column_names(dtypes=[str, int, float])) == 13
    assert len(df.get_column_names(dtypes=[str, int, float, bool])) == len(df.get_column_names()) == 14
    # TODO datetime


def test_getitem_dtypes():
    df = load_titanic()
    assert df[int].shape[1] == 3
    assert df[[int]].shape[1] == 3
    assert df[tuple([int])].shape[1] == 3
    assert df[float].shape[1] == 3
    assert df[str].shape[1] == 7
    assert df[object].shape[1] == 7
    assert df[bool].shape[1] == 1
    assert df[str, int].shape[1] == 10
    assert df[str, int, float].shape[1] == 13
    assert df[str, int, float, bool].shape[1] == 14 == len(df.get_column_names())
    # TODO datetime

import pyarrow as pa
import vaex.groupby



def test_combined_grouper():
    x = pa.array([1, 1, 2, 2])
    y = pa.array([1, 1, 3, 4])
    df = vaex.from_arrays(x=x, y=y)
    grouper_x = vaex.groupby.Grouper(df.x)
    grouper_y = vaex.groupby.Grouper(df.y)
    groupers = [grouper_x, grouper_y]
    assert set(df[grouper_x.binby_expression].tolist()) == {0, 1}
    assert set(df[grouper_y.binby_expression].tolist()) == {0, 1, 2}
    grouper = vaex.groupby._combine(df, groupers, sort=True)
    assert set(df[grouper.binby_expression].tolist()) == {0, 1, 2}


def test_combined_grouper_cast():
    x = pa.array(range(127))
    y = pa.array(range(1, 128))
    df = vaex.from_arrays(x=x, y=y)
    grouper_x = vaex.groupby.Grouper(df.x)
    grouper_y = vaex.groupby.Grouper(df.y)
    groupers = [grouper_x, grouper_y]
    assert df[grouper_x.binby_expression].dtype.numpy.name == 'int8'
    assert df[grouper_y.binby_expression].dtype.numpy.name == 'int8'
    grouper = vaex.groupby._combine(df, groupers, sort=True)
    assert set(df[grouper.binby_expression].tolist()) == set(range(127))

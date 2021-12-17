from unittest import mock
from common import *


@pytest.mark.parametrize("chunk_size", list(range(1, 13)))
@pytest.mark.parametrize("prefetch", [True, False])
@pytest.mark.parametrize("parallel", [True, False])
def test_evaluate_iterator(df_local, chunk_size, prefetch, parallel):
    df = df_local
    with small_buffer(df):
        x = df.x.to_numpy()
        z = df.z.to_numpy()
        total = 0
        for i1, i2, chunk in df_local.evaluate_iterator('x', chunk_size=chunk_size, prefetch=prefetch,
                                                        parallel=parallel, array_type='numpy-arrow'):
            assert x[i1:i2].tolist() == chunk.tolist()
            total += chunk.sum()
        assert total == x.sum()

        total = 0
        for i1, i2, chunk in df_local.evaluate_iterator('z', chunk_size=chunk_size, prefetch=prefetch,
                                                        parallel=parallel, array_type='numpy-arrow'):
            assert z[i1:i2].tolist() == chunk.tolist()
            total += chunk.sum()
        assert total == z.sum()


@pytest.mark.parametrize("chunk_size", [2, 5])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("array_type", ['numpy', 'list', 'xarray'])
def test_to_items(df_local, chunk_size, parallel, array_type):
    df = df_local
    x = df.x.to_numpy()
    total = 0
    for i1, i2, chunk in df.to_items(['x'], chunk_size=chunk_size, parallel=parallel, array_type=array_type):
        np.testing.assert_array_equal(x[i1:i2], chunk[0][1])
        total += sum(chunk[0][1])
    assert total == x.sum()


@pytest.mark.parametrize("chunk_size", [2, 5])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("array_type", ['numpy', 'list', 'xarray'])
def test_to_dict(df_local, chunk_size, parallel, array_type):
    df = df_local
    x = df.x.to_numpy()
    total = 0
    for i1, i2, chunk in df.to_dict(['x'], chunk_size=chunk_size, parallel=parallel, array_type=array_type):
        np.testing.assert_array_equal(x[i1:i2], chunk['x'])
        total += sum(chunk['x'])
    assert total == x.sum()


@pytest.mark.parametrize("chunk_size", [2, 5])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("array_type", ['numpy', 'list', 'python'])
def test_to_records(df_local, chunk_size, parallel, array_type):
    df = df_local
    for i1, i2, chunk in df.to_records(chunk_size=chunk_size, parallel=parallel, array_type=array_type):
        assert isinstance(chunk, list)
        assert len(chunk) <= chunk_size
        assert isinstance(chunk[0], dict)
    record = df.to_records(0)
    assert isinstance(record, dict)
    assert isinstance(df.to_records(chunk_size=None, parallel=parallel, array_type=array_type), list)


@pytest.mark.parametrize("chunk_size", [2, 5])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("array_type", ['numpy', 'list', 'xarray'])
def test_to_arrays(df_local, chunk_size, parallel, array_type):
    df = df_local
    x = df.x.to_numpy()
    total = 0
    for i1, i2, chunk in df.to_arrays(['x'], chunk_size=chunk_size, parallel=parallel, array_type=array_type):
        np.testing.assert_array_equal(x[i1:i2], chunk[0])
        total += sum(chunk[0])
    assert total == x.sum()


def test_evaluate_function_filtered_df():
    # Custom function to be applied to a filtered DataFrame
    def custom_func(x):
        assert 4 not in x;
        return x ** 2

    df = vaex.from_arrays(x=np.arange(10))
    df_filtered = df[df.x != 4]
    df_filtered.add_function('custom_function', custom_func)
    df_filtered['y'] = df_filtered.func.custom_function(df_filtered.x)
    assert df_filtered.y.tolist() == [0, 1, 4, 9, 25, 36, 49, 64, 81]

    # sliced exactly at the start of where we are going to filter
    # this used to trigger a bug in df.dtype, which would evaluate the first row
    df_sliced = df[4:]
    df_filtered = df_sliced[df_sliced.x != 4]
    df_filtered.add_function('custom_function', custom_func)
    df_filtered['y'] = df_filtered.func.custom_function(df_filtered.x)
    assert df_filtered.y.tolist() == [25, 36, 49, 64, 81]


def test_bool(df_trimmed):
    df = df_trimmed
    expr = df.x * df.y
    assert bool(expr == expr)
    assert not bool(expr != expr)
    assert bool(expr != expr + 1)
    assert not bool(expr == None)


def test_aliased():
    df = vaex.from_dict({'1': [1, 2], '#': [2, 3]})
    assert df.evaluate('#').tolist() == [2, 3]


def test_evaluate_types():
    x = np.arange(2)
    y = pa.array(x ** 2)
    s = pa.array(["foo", "bars"])
    df = vaex.from_arrays(x=x, y=y, s=s)
    assert isinstance(df.columns['x'], np.ndarray)
    assert isinstance(df.columns['y'], pa.Array)

    assert df.evaluate("x", array_type=None) is x
    assert isinstance(df.evaluate("x", array_type="numpy"), np.ndarray)
    assert isinstance(df.evaluate("x", array_type="arrow"), pa.Array)

    # TODO: we want to add real string arrays only, so it should be arrow
    assert df.evaluate("s", array_type=None) is s
    assert isinstance(df.evaluate("s", array_type=None), pa.Array)
    assert isinstance(df.evaluate("s", array_type="arrow"), pa.Array)
    assert isinstance(df.evaluate("s", array_type="numpy"), np.ndarray)

    assert df.evaluate("y", array_type=None) is y
    assert isinstance(df.evaluate("y", array_type=None), pa.Array)
    assert isinstance(df.evaluate("y", array_type="arrow"), pa.Array)
    assert isinstance(df.evaluate("y", array_type="numpy"), np.ndarray)


@pytest.mark.parametrize('parallel', [True, False])
def test_arrow_evaluate(parallel):
    x = np.arange(2)
    l = pa.array([[1, 2], [2, 3, 4]])
    df = vaex.from_arrays(s=["foo", "bars"], l=l)
    assert df.evaluate(df.s.as_arrow(), array_type='numpy', parallel=parallel).dtype == object
    assert df.evaluate(df.s.as_arrow(), array_type='arrow', parallel=parallel).type == pa.string()
    assert df.evaluate(df.s.as_arrow(), array_type=None, parallel=parallel).type == pa.string()
    assert df.evaluate(df.l, parallel=parallel).type == pa.list_(l.type.value_type)


def test_evaluate_with_selection(df_factory):
    x = np.arange(3)
    df = df_factory(x=x)
    assert df.x.evaluate(selection='x>0', array_type='numpy').tolist() == [1, 2]
    assert df.x.evaluate(selection='x>0', array_type='arrow').to_pylist() == [1, 2]
    assert df.x.evaluate(selection='x>0', array_type='python') == [1, 2]


@pytest.mark.parametrize("array_type", ['numpy', 'list', 'arrow'])
def test_evaluate_chunked(df_factory, buffer_size, array_type):
    x = np.arange(10)
    y = x**2
    df = df_factory(x=x)
    # use a virtual column, since 'x' skips passing over the data
    df['y'] = df.x ** 2
    with buffer_size(df, 3):
        values = df.evaluate('y', array_type=array_type)
        assert vaex.array_types.tolist(values) == y.tolist()

def test_evaluate_no_execute():
    df = vaex.from_dict({"#": [1.1], "with space": ['should work'], "x": [1.]})
    df['%'] = df['#'] + 1
    with mock.patch.object(df.executor, 'execute', wraps=df.executor.execute) as method:
        df.evaluate('x')
        method.assert_not_called()
        df.evaluate('df["#"]')
        method.assert_not_called()
        df.evaluate(df["#"])
        method.assert_not_called()
        df.evaluate(df["%"])
        method.assert_called_once()


@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize("prefetch", [True, False])
def test_evaluate_empty(parallel, prefetch):
    df = vaex.from_arrays(x=[1, 2])
    dff = df[df.x > 10]
    assert dff.x.tolist() == []
    assert dff.evaluate('x', parallel=parallel).tolist() == []
    for i1, i2, chunks in dff.evaluate_iterator('x', chunk_size=10, parallel=parallel, prefetch=prefetch):
        raise RuntimeError('unexpected')
    for i1, i2, chunks in dff.to_arrow_table('x', chunk_size=10, parallel=parallel):
        raise RuntimeError('unexpected')

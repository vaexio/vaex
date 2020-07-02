import pytest
import vaex
import pyarrow as pa

@pytest.mark.parametrize('op', vaex.expression._binary_ops)
def test_binary_ops(df_trimmed, op):
    if op['name'] in ['contains', 'and', 'xor', 'or', 'is', 'is_not', 'matmul']:
        return
    
    operator = op['op']
    df = df_trimmed
    x = df.x.astype(pa.int32()).to_numpy()
    y = df.y.astype(pa.int64()).to_numpy()
    df['x'] = df.x.astype(pa.int32()).as_arrow()
    df['y'] = df.y.astype(pa.int64()).as_arrow()
    df['pa'] = operator(df.x, 1 + df.y)
    assert df.pa.tolist() == operator(x, 1 + y).tolist()
    assert df.pa.values.to_pylist() == operator(x, 1 + y).tolist()

import vaex.operations as vop
import vaex

def test_open():
    path = vaex.example().path
    op = vaex.open(path, execute=False)
    df = op.execute()
    assert df.path == path
    assert df.operation.execute().path == path


def test_add_virtual_column():
    path = vaex.example().path
    op = vaex.open(path, execute=False)
    df = op.execute()
    df['r'] = str(df.x + df.y)
    assert df.operation.name == 'add_virtual_column'
    assert df.operation.args[0] == 'r'
    assert df.operation.child is op
    
    df['r'] = (df.r + df.y)
    assert df.operation.name == 'add_virtual_column'
    assert df.operation.child.name == 'rename_column'
    repr(df.operation)

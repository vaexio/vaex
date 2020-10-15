import vaex
import numpy as np

def test_expression_expand():
    ds = vaex.from_scalars(x=1, y=2)

    ds['g'] = ds.x
    assert ds.g.expression == 'g'
    assert ds.g.variables() == {'x'}
    # TODO: this doesn't work, because outself and include_virtual contradict eachother
    # but we don't use this interally
    # assert ds.g.variables(ourself=True, include_virtual=False) == {'g', 'x'}

    ds['r'] = ds.x * ds.y
    assert ds.r.expression == 'r'
    assert ds.r.variables() == {'x', 'y'}
    assert ds.r.variables(ourself=True, include_virtual=False) == {'r', 'x', 'y'}
    ds['s'] = ds.r + ds.x
    assert ds.s.variables() == {'r', 'x', 'y'}
    assert ds.s.variables(ourself=True) == {'s', 'r', 'x', 'y'}
    assert ds.s.variables(include_virtual=False) == {'x', 'y'}
    assert ds.s.variables(ourself=True, include_virtual=False) == {'s', 'x', 'y'}
    ds['t'] = ds.s + ds.y
    assert ds.t.variables() == {'s', 'r', 'x', 'y'}
    ds['u'] = np.arctan(ds.t)
    assert ds.u.variables() == {'t', 's', 'r', 'x', 'y'}


def test_non_identifiers():
    df = vaex.from_dict({'x': [1], 'y': [2], '#':[1]})
    df['z'] = df['#'] + 1
    assert df['z'].variables() == {'#'}
    assert df._virtual_expressions['z'].variables() == {'#'}

    df['1'] = df.x * df.y
    df['2'] = df['1'] + df.x
    assert df['1'].variables(ourself=True) == {'x', 'y', '1'}
    assert df['1'].variables() == {'x', 'y'}
    assert df['2'].variables(ourself=True) == {'x', 'y', '2', '1'}
    assert df['2'].variables(include_virtual=False) == {'x', 'y'}

    df['valid'] = df['2']
    assert df['valid'].variables(ourself=True) == {'x', 'y', '2', '1', 'valid'}
    assert df['valid'].variables(include_virtual=False) == {'x', 'y'}

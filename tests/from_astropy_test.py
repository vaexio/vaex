import warnings

import numpy as np
import pytest
from astropy.table import Table

import vaex


@pytest.mark.parametrize('shape', [(3, 2), (3, 2, 2)])
@pytest.mark.parametrize('dtype', ['float64', 'int32', 'uint32', 'S4', 'U4'])
def test_from_astropy_warns_about_multidimensional_columns(shape, dtype):
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    table = Table({'before': [10, 20, 30], 'multidimensional': data, 'after': [1., 2., 3.]})

    with pytest.warns(UserWarning, match='multidimensional') as caught:
        df = vaex.from_astropy_table(table)

    assert len(caught) == 1
    assert caught[0].filename == __file__
    assert df.get_column_names() == ['before', 'after']
    assert df.before.tolist() == [10, 20, 30]
    assert df.after.tolist() == [1., 2., 3.]


def test_from_astropy_warns_for_each_dropped_column():
    table = Table({'scalar': [1, 2], 'vectors': [[1, 2], [3, 4]],
                   'matrices': [[[1]], [[2]]]})

    with pytest.warns(UserWarning) as caught:
        df = vaex.from_astropy_table(table)

    assert len(caught) == 2
    assert 'vectors' in str(caught[0].message)
    assert 'matrices' in str(caught[1].message)
    assert df.scalar.tolist() == [1, 2]


def test_from_astropy_scalar_columns_do_not_warn():
    table = Table({'number': [1, 2], 'text': ['a', 'b']})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        df = vaex.from_astropy_table(table)

    assert not caught
    assert df.number.tolist() == [1, 2]
    assert df.text.tolist() == ['a', 'b']

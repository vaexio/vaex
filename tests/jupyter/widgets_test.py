import vaex.jupyter.traitlets as vt
import vaex
import numpy as np

def test_column_list_traitlets():
    df = vaex.from_scalars(x=1, y=2)
    df['z'] = df.x + df.y
    column_list = vt.ColumnsMixin(df=df)
    assert len(column_list.columns) == 3
    df['w'] = df.z * 2
    assert len(column_list.columns) == 4
    del df['w']
    assert len(column_list.columns) == 3

import vaex
import numpy as np
from functools import reduce

def test_concat():
    x1, y1, z1 = np.arange(3), np.arange(3,0,-1), np.arange(10,13)
    x2, y2, z2 = np.arange(3,6), np.arange(0,-3,-1), np.arange(13,16)
    x3, y3, z3 = np.arange(6,9), np.arange(-3,-6,-1), np.arange(16,19)
    w1, w2, w3 = np.array(['cat']*3), np.array(['dog']*3), np.array(['fish']*3)
    x = np.concatenate((x1,x2,x3))
    y = np.concatenate((y1,y2,y3))
    z = np.concatenate((z1,z2,z3))
    w = np.concatenate((w1,w2,w3))

    ds  = vaex.from_arrays(x=x, y=y, z=z, w=w)
    ds1 = vaex.from_arrays(x=x1, y=y1, z=z1, w=w1)
    ds2 = vaex.from_arrays(x=x2, y=y2, z=z2, w=w2)
    ds3 = vaex.from_arrays(x=x3, y=y3, z=z3, w=w3)

    dd = vaex.concat([ds1, ds2])
    ww = ds1.concat(ds2)

    # Test if the concatination of two arrays with the vaex method is the same as with the dataset method
    assert (np.array(dd.evaluate('x,y,z,w')) == np.array(ww.evaluate('x,y,z,w'))).all()

    # Test if the concatination of multiple datasets works
    dd = vaex.concat([ds1, ds2, ds3])
    assert (np.array(dd.evaluate('x,y,z,w')) == np.array(ds.evaluate('x,y,z,w'))).all()

    # Test if the concatination of concatinated datasets works
    dd1 = vaex.concat([ds1, ds2])
    dd2 = vaex.concat([dd1, ds3])
    assert (np.array(dd2.evaluate('x,y,z,w')) == np.array(ds.evaluate('x,y,z,w'))).all()
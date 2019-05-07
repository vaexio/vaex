from common import *


def test_count_obj(ds_local):
    df = ds_local
    # df.count('x')
    df.count('obj', delay=True)
    # dsa

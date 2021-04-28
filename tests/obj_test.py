from common import *


def test_count_obj(df_local_non_arrow):
    df = df_local_non_arrow
    df.count('obj', delay=True)

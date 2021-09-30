

def test_stack_basic(df_factory):
    df = df_factory(x=[1,2,3], y=[2,3,4])
    df['z'] = df.func.stack([df.x, df.y])
    assert df.z.tolist() == [[1,2], [2,3], [3,4]]


def test_stack_missing(df_factory):
    df = df_factory(x=[1,None,3, None], y=[2,3,4,None])
    df['z'] = df.func.stack([df.x, df.y])
    assert df.z.tolist() == [[1,2], [None,3], [3,4], [None, None]]

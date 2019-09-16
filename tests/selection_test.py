import numpy as np

import vaex


def test_selection_and_filter():
    x = np.arange(-10, 11, 1)
    y = np.arange(21)
    df = vaex.from_arrays(x=x, y=y)

    df.select(df.x < 0)
    selected_list = df.evaluate(df.x, selection=True).tolist()

    df_filtered = df[df.x < 0]
    filtered_list = df_filtered['x'].tolist()
    assert filtered_list == selected_list

    # make sure we can slice, and repr
    df_sliced = df_filtered[:5]
    repr(df_filtered)

from common import *

def test_trim(ds_local):
    ds = ds_local
    ds_trimmed = ds.trim()
    assert ds_trimmed.length_original() == ds_trimmed.length_unfiltered() >= 10
    assert ds_trimmed.get_active_range() == (0, ds_trimmed.length_original())# == (0, 10)
    assert ds_trimmed.evaluate('x').tolist() == np.arange(10.).tolist()

    # trimming with a non-zero start index
    start = ds.get_active_range()[0]
    if ds.filtered:
        start = 2  # dirty hack
    ds.set_active_range(start+5, start+10)
    ds_trimmed = ds.trim()
    assert ds_trimmed.length_original() == ds_trimmed.length_unfiltered() == 5
    assert ds_trimmed.get_active_range() == (0, ds_trimmed.length_original()) == (0, 5)
    assert ds_trimmed.evaluate('x').tolist() == np.arange(5, 10.).tolist()

    # trim on trimmed
    ds_trimmed.set_active_range(1, 4)
    ds_trimmed = ds_trimmed.trim()

    assert ds_trimmed.length_original() == ds_trimmed.length_unfiltered() == 3
    assert ds_trimmed.get_active_range() == (0, ds_trimmed.length_original()) == (0, 3)
    assert ds_trimmed.evaluate('x').tolist() == np.arange(6, 9.).tolist()


def test_trim_hidden(df_local_non_arrow):
    df = df_local_non_arrow
    df['r'] = df.x + df.y
    df_sub = df[['r']].head()
    assert len(df_sub.columns['__x']) == len(df_sub)

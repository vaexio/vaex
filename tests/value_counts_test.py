from common import create_base_ds


def test_value_counts():
    ds = create_base_ds()

    v_counts = ds.x.value_counts()
    assert len(v_counts) == 21

    v_counts = ds.y.value_counts()
    assert len(v_counts) == 19

    v_counts = ds.m.value_counts()
    assert len(v_counts) == 20

    v_counts = ds.n.value_counts()
    assert len(v_counts) == 20

    v_counts = ds.nm.value_counts()
    assert len(v_counts) == 19

    v_counts = ds.mi.value_counts()
    assert len(v_counts) == 20

    v_counts = ds.obj.value_counts()
    assert len(v_counts) == 19

    v_counts_name = ds['name'].value_counts()
    v_counts_name_arrow = ds.name_arrow.value_counts()
    assert np.all(v_counts_name == v_counts_name_arrow)

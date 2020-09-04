import vaex


def test_ipython_autocompletion(ds_local):
    df = vaex.from_dict({'First name': ['Reggie', 'Tamika'],
                         'Last name': ['Miller', 'Catchings'],
                         '$amount': [10, 20]})

    completions = df._ipython_key_completions_()
    assert 'First name' in completions
    assert 'Last name' in completions
    assert '$amount' in completions
    assert 'Team' not in completions

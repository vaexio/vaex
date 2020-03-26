def test_count_huge(client):
    # this catched a bug introduced in https://github.com/vaexio/vaex/pull/557
    # where remote dataframe calculations were cancelled
    df = client['huge']
    assert df.count() == len(df)

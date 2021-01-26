# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
import vaex


def test_nan_madness():
    nan = float('NaN')
    x = [np.nan, nan, np.nan/2, nan/3, 0, 1]
    df = vaex.from_arrays(x=x)
    mapper = {np.nan/5: -1, 0: 10, 1: 20}
    assert df.x.map(mapper).tolist() == [-1, -1, -1, -1, 10, 20]

    mapper = {np.nan/5: -1, np.nan/10:-2, 0: 10, 1: 20}
    with pytest.raises(ValueError):
        df.x.map(mapper).tolist()


def test_map_basics():
    # Generate the test data
    colour = ['red', 'red', 'blue', 'red', 'green', 'green', 'red', 'blue', 'blue', 'green']
    animal = np.array(['dog', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'dog', 'dog', np.nan], dtype='O')
    number = [10, 20, 30, 10, 20, 30, 30, 30, 10, 20]
    floats = [10., 20., 30., 10., 20., 30., 30., 30., 10., np.nan]
    ds = vaex.from_arrays(colour=colour, animal=animal, number=number, floats=floats)
    df = pd.DataFrame(data=np.array([colour, animal, number, floats]).T, columns=['colour', 'animal', 'number', 'floats'])

    # Create a mapper - dictionary
    mapper = {}
    mapper['colour'] = {'red': 1, 'blue': 2, 'green': 3}
    mapper['animal'] = {'dog': 5, 'cat': -1, 'dolphin': 0}
    mapper['number'] = {10: 1, 20: 2, 30: 3}
    mapper['floats'] = {10.: -1, 20.: -2, 30.: -3, np.nan: -4}

    # Map the functions in vaex
    ds['colour_'] = ds.colour.map(mapper['colour'])
    ds['animal_'] = ds.animal.map(mapper['animal'])
    # ds['number_'] = ds.number.map(lambda x: mapper['number'][x])  # test with a function, not just with a dict
    ds['floats_'] = ds.floats.map(mapper['floats'], nan_value=np.nan)

    # Map in pandas
    df['colour_'] = df.colour.map(mapper['colour'])
    df['animal_'] = df.animal.map(mapper['animal'])

    # Make assertions - compare to pandas for string columns
    # we deviate from pandas, we can map nan to something
    assert ds.colour_.values.tolist()[:-1] == df.colour_.values.tolist()[:-1]
    assert ds.animal_.values.tolist()[:-1] == df.animal_.values.tolist()[:-1]
    assert ds.animal_.values.tolist()[-1] is None
    # Make assertions - compare to the expected values for numeric type
    # assert ds.number_.values.tolist() == (np.array(number)/10).tolist()
    assert ds.floats_.values.tolist()[:-1] == (np.array(floats)/-10.).tolist()[:-1]
    assert ds.floats_.values.tolist()[-1] == -4

    # missing keys
    with pytest.raises(ValueError):
        ds.colour.map({'ret': 1, 'blue': 2, 'green': 3})
    with pytest.raises(ValueError):
        ds.colour.map({'blue': 2, 'green': 3})
    # missing keys but user-handled
    ds['colour_unmapped'] = ds.colour.map({'blue': 2, 'green': 3}, default_value=-1)
    assert ds.colour_unmapped.values.tolist() == [-1, -1, 2, -1, 3, 3, -1, 2, 2, 3]
    # extra is ok
    ds.colour.map({'red': 1, 'blue': 2, 'green': 3, 'orange': 4})

    # check masked arrays
    # import pdb; pdb.set_trace()
    assert ds.colour.map({'blue': 2, 'green': 3}, allow_missing=True).tolist() == [None, None, 2, None, 3, 3, None, 2, 2, 3]


def test_map_missing(df_factory):
    df = df_factory(x=[1, 2, None])
    df['m'] = df.x.map({1: 99}, allow_missing=True)
    assert df.m.dtype == int
    assert df.m.tolist() == [99, None, None]


def test_map_to_string():
    df = vaex.from_arrays(type=[0, 1, 2, 2, 2, np.nan])
    df['role'] = df['type'].map({0: 'admin', 1: 'maintainer', 2: 'user', np.nan: 'unknown'})
    assert df['role'].tolist() == ['admin', 'maintainer', 'user', 'user', 'user', 'unknown']


@pytest.mark.parametrize("type", [pa.string(), pa.large_string()])
def test_map_from_string(type):
    df = vaex.from_arrays(type=pa.array(['admin', 'maintainer', 'user', 'user', 'user', None], type=type))
    df['role'] = df['type'].map({'admin':0, 'maintainer':1, 'user':2, None: -1})
    assert df['role'].tolist() == [0, 1, 2, 2, 2, -1]


def test_map_serialize(tmpdir):
    df = vaex.from_arrays(type=[0, 1, 2, 2, 2, np.nan])
    df['role'] = df['type'].map({0: 'admin', 1: 'maintainer', 2: 'user', np.nan: 'unknown'})
    assert df['role'].tolist() == ['admin', 'maintainer', 'user', 'user', 'user', 'unknown']
    path = str(tmpdir.join('state.json'))
    df.state_write(path)

    df = vaex.from_arrays(type=[0, 1, 2, 2, 2, np.nan])
    df.state_load(path)
    assert df['role'].tolist() == ['admin', 'maintainer', 'user', 'user', 'user', 'unknown']

def test_map_serialize_string(tmpdir):
    df = vaex.from_arrays(type=['0', '1', '2', '2', '2'])
    df['role'] = df['type'].map({'0': 'admin', '1': 'maintainer', '2': 'user'})
    assert df['role'].tolist() == ['admin', 'maintainer', 'user', 'user', 'user']
    path = str(tmpdir.join('state.json'))
    df.state_write(path)

    df = vaex.from_arrays(type=['0', '1', '2', '2', '2'])
    df.state_load(path)
    assert df['role'].tolist() == ['admin', 'maintainer', 'user', 'user', 'user']


def test_map_long_mapper():
    german = np.array(['eins', 'zwei', 'drei', 'vier', 'fünf', 'sechs', 'sieben', 'acht', 'neun', 'zehn', 'elf', 'zwölf',
                       'dreizehn', 'vierzehn', 'fünfzehn', 'sechzehn', 'siebzehn', 'achtzehn', 'neunzehn', 'zwanzig',
                       'einundzwanzig', 'zweiundzwanzig', 'dreiundzwanzig', 'vierundzwanzig', 'fünfundzwanzig', 'sechsundzwanzig',
                       'siebenundzwanzig', 'achtundzwanzig', 'neunundzwanzig', 'dreiβig', 'einunddreiβig', 'zweiunddreißig',
                       'dreiunddreißig', 'vierunddreißig', 'fünfunddreißig', 'sechsunddreißig', 'siebenunddreißig',
                       'achtunddreißig', 'neununddreißig', 'vierzig', 'einundvierzig', 'zweiundvierzig', 'dreiundvierzig',
                       'vierundvierzig', 'fünfundvierzig', 'sechsundvierzig', 'siebenundvierzig', 'achtundvierzig',
                       'neunundvierzig', 'fünfzig'])

    english = np.array(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve',
                        'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen', 'twenty', 'twentyone',
                        'twentytwo', 'twentythree', 'twentyfour', 'twentyfive', 'twentysix', 'twentyseven', 'twentyeight',
                        'twentynine', 'thirty', 'thirtyone', 'thirtytwo', 'thirtythree', 'thirtyfour', 'thirtyfive', 'thirtysix',
                        'thirtyseven', 'thirtyeight', 'thirtynine', 'forty', 'fortyone', 'fortytwo', 'fortythree', 'fortyfour',
                        'fortyfive', 'fortysix', 'fortyseven', 'fortyeight', 'fortynine', 'fifty'])

    mapper = dict(zip(english, german))  # enlish to german
    df = vaex.from_arrays(english=english)
    df['german'] = df.english.map(mapper=mapper)
    assert df['german'].tolist() == german.tolist()



def test_unique_list(df_types):
    df = df_types
    mapper = {'aap': 1, 'noot': 2, 'mies': 3, None: 999}
    expected = [[mapper[el] for el in list] if list is not None else None for list in df.string_list.tolist()]

    assert df.string_list.map(mapper).tolist() == expected
    assert set(df.int_list.unique()) == {1, 2, 3, 4, 5, None}

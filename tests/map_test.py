import numpy as np

import pandas as pd

import pytest

import vaex


def test_map():
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
    assert ds.animal_.values[-1] is None
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
    assert ds.colour.map({'blue': 2, 'green': 3}, allow_missing=True).tolist() == [None, None, 2, None, 3, 3, None, 2, 2, 3]


def test_map_to_string():
    df = vaex.from_arrays(type=[0, 1, 2, 2, 2, np.nan])
    df['role'] = df['type'].map({0: 'admin', 1: 'maintainer', 2: 'user', np.nan: 'unknown'})
    assert df['role'].tolist() == ['admin', 'maintainer', 'user', 'user', 'user', 'unknown']


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

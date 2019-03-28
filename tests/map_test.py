import vaex
import numpy as np
import pandas as pd


def test_map():
    # Generate the test data
    colour = ['red', 'red', 'blue', 'red', 'green', 'green', 'red', 'blue', 'blue', 'green']
    animal = ['dog', 'cat', 'dog', 'dog', 'dog', 'dog', 'cat', 'dog', 'dog', np.nan]
    number = [10, 20, 30, 10, 20, 30, 30, 30, 10, 20]
    floats = [10., 20., 30., 10., 20., 30., 30., 30., 10., np.nan]
    ds = vaex.from_arrays(colour=colour, animal=animal, number=number, floats=floats)
    df = pd.DataFrame(data=np.array([colour, animal, number, floats]).T, columns=['colour', 'animal', 'number', 'floats'])

    # Create a mapper - dictionary
    mapper = {}
    mapper['colour'] = {'red': 1, 'blue': 2, 'green': 3}
    mapper['animal'] = {'dog': 5, 'cat': -1, 'dolphin': 0}
    mapper['number'] = {10: 1, 20: 2, 30: 3}
    mapper['floats'] = {10.: -1, 20.: -2, 30.: -3}

    # Map the functions in vaex
    ds['colour_'] = ds.colour.map(mapper['colour'])
    ds['animal_'] = ds.animal.map(mapper['animal'])
    ds['number_'] = ds.number.map(lambda x: mapper['number'][x])  # test with a function, not just with a dict
    ds['floats_'] = ds.floats.map(mapper['floats'], missing_values=np.nan)

    # Map in pandas
    df['colour_'] = df.colour.map(mapper['colour'])
    df['animal_'] = df.animal.map(mapper['animal'])

    # Make assertions - compare to pandas for string columns
    assert ds.colour_.values.tolist() == df.colour_.values.tolist()
    assert ds.animal_.values.tolist()[:-1] == df.animal_.values.tolist()[:-1]
    assert ds.animal_.values[-1] is None
    # Make assertions - compare to the expected values for numeric type
    assert ds.number_.values.tolist() == (np.array(number)/10).tolist()
    assert ds.floats_.values.tolist()[:-1] == (np.array(floats)/-10.).tolist()[:-1]
    assert np.isnan(ds.floats_.values[-1])

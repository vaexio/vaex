import json
import os

import vaex


def test_export_json():
    temp_path = './temp.json'
    ds = vaex.from_arrays(**{'A': [1, 2, 3], 'B': ['1', '2', '3']})
    vaex.utils.write_json_or_yaml(temp_path, ds.to_dict(array_type='python'))

    with open(temp_path, 'r') as f:
        data = json.load(f)

    os.remove(temp_path)
    assert 'A' in data
    assert len(data['A']) == 3
    assert data['B'][0] == '1'

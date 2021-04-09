from common import *
from io import BytesIO
import pickle

import pytest

import vaex.encoding
import vaex.dataset


def rebuild_dataset_pickle(ds):
    # pick and unpickle
    f = BytesIO()
    picked = pickle.dump(ds, f)
    f.seek(0)
    return pickle.load(f)


def rebuild_dataset_vaex(ds):
    # encoding and decode
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('dataset', ds)
    blob = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(blob, encoding)
    return encoding.decode('dataset', data)


@pytest.fixture(params=['rebuild_dataset_pickle', 'rebuild_dataset_vaex'])
def rebuild_dataset(request):
    named = dict(rebuild_dataset_pickle=rebuild_dataset_pickle, rebuild_dataset_vaex=rebuild_dataset_vaex)
    return named[request.param]

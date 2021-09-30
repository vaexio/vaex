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

@pytest.fixture
def repickle():
    def wrapper(obj):
        f = BytesIO()
        picked = pickle.dump(obj, f)
        f.seek(0)
        return pickle.load(f)
    return wrapper



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

# similar, but for dataframe
rebuild_dataframe_pickle = rebuild_dataset_pickle


def rebuild_dataframe_vaex(df):
    # encoding and decode
    encoding = vaex.encoding.Encoding()
    data = encoding.encode('dataframe', df)
    blob = vaex.encoding.serialize(data, encoding)

    encoding = vaex.encoding.Encoding()
    data = vaex.encoding.deserialize(blob, encoding)
    return encoding.decode('dataframe', data)


@pytest.fixture(params=['rebuild_dataframe_pickle', 'rebuild_dataframe_vaex'])
def rebuild_dataframe(request):
    named = dict(rebuild_dataframe_pickle=rebuild_dataframe_pickle, rebuild_dataframe_vaex=rebuild_dataframe_vaex)
    return named[request.param]


@pytest.fixture
def no_vaex_cache():
    with vaex.cache.off():
        yield

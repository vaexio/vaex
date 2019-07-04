import numpy as np
import vaex
import vaex.webserver
import pytest

import sys
test_port = 29110 + sys.version_info[0] * 10 + sys.version_info[1]
scheme = 'ws'

@pytest.fixture(scope='module')
def webserver():
    webserver = vaex.webserver.WebServer(datasets=[], port=test_port, cache_byte_size=0, token='token', token_trusted='token_trusted')
    x = np.arange(10)
    df = vaex.from_arrays(x=x)
    df.name = 'df'
    webserver.set_datasets([df])
    webserver.serve_threaded()
    yield webserver
    webserver.stop_serving()
    #return webserver

@pytest.fixture(scope='module')
def server(webserver):
    server = vaex.server("%s://localhost:%d" % (scheme, test_port))
    yield server
    server.close()

@pytest.fixture()
def base_url():
    return "%s://localhost:%d" % (scheme, test_port)
# @pytest.fixture()
# def df_remote(webserver, server):
#     return server.datasets(as_dict=True)['df']

def test_no_auth(webserver, base_url):
    with pytest.raises(ValueError, match='.*No token.*'):
        df = vaex.open("%s/df" % (base_url))
        df.x.sum()
        pytest.fail('When no token given, operations should not be supported')


def test_no_trusted(webserver, base_url):
    df = vaex.open("%s/df?token=token" % (base_url))
    df.x.sum()
    df.x.jit_numba().sum()
    with pytest.raises(ValueError, match='.*pickle.*'):
        df.x.apply(lambda x: x+1).sum()
        pytest.fail('When no token_trusted given, function serialization operations using pickle should not be supported')


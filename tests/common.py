import pytest
import vaex
import vaex.server.service
import vaex.server.tornado_server
import vaex.server.dummy
import numpy as np
import contextlib
import pyarrow as pa

import sys
test_port = 3911 + sys.version_info[0] * 10 + sys.version_info[1]
scheme = 'ws'


class CallbackCounter(object):
    def __init__(self, return_value=None):
        self.counter = 0
        self.return_value = return_value
        self.last_args = None
        self.last_kwargs = None

    def __call__(self, *args, **kwargs):
        self.counter += 1
        self.last_args = args
        self.last_kwargs = kwargs
        return self.return_value


@contextlib.contextmanager
def small_buffer(ds, size=3):
    if ds.is_local():
        previous = ds.executor.buffer_size
        ds.executor.buffer_size = size
        ds._invalidate_selection_cache()
        try:
            yield
        finally:
            ds.executor.buffer_size = previous
    else:
        yield # for remote datasets we don't support this ... or should we?


@pytest.fixture(scope='session')
def webserver():
    webserver = vaex.server.tornado_server.WebServer(datasets=[], port=test_port, cache_byte_size=0)
    webserver.serve_threaded()
    yield webserver
    webserver.stop_serving()
    #return webserver


# the dataframe that lives at the server
@pytest.fixture()
def df_server(ds_trimmed):
    df = ds_trimmed.copy()
    df.name = 'test'
    return df


@pytest.fixture()
def df_server_huge():
    df = vaex.from_arrays(x=vaex.vrange(0, int(1e9)))
    df.name = 'huge'
    return df


# as in https://github.com/erdewit/nest_asyncio/issues/20
@pytest.fixture(scope="session")
def event_loop():
    """Don't close event loop at the end of every function decorated by
    @pytest.mark.asyncio
    """
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()


@pytest.fixture()#scope='module')
def tornado_client(webserver, df_server, df_server_huge, event_loop):
    df = df_server
    df.drop('obj', inplace=True)
    df.drop('datetime', inplace=True)
    df.drop('timedelta', inplace=True)
    webserver.set_datasets([df, df_server_huge])
    client = vaex.connect("%s://localhost:%d" % (scheme, test_port))
    yield client
    client.close()


@pytest.fixture
def dummy_client(df_server, df_server_huge):
    df = df_server
    service = vaex.server.service.Service({'test': df, 'huge': df_server_huge})
    server = vaex.server.dummy.Server(service)
    client = vaex.server.dummy.Client(server)
    return client


# @pytest.fixture(params=['dummy_client', 'tornado_client'])
@pytest.fixture(params=['tornado_client'])
def client(request, dummy_client, tornado_client):
    named = dict(dummy_client=dummy_client, tornado_client=tornado_client)
    return named[request.param]


@pytest.fixture()
def ds_remote(client):
    return client['test']


@pytest.fixture()
def df_remote(ds_remote):
    return ds_remote


@pytest.fixture()
def ds_filtered(df_filtered):
    return df_filtered


@pytest.fixture()
def df_filtered():
    return create_filtered()

@pytest.fixture()
def ds_half():
    ds = create_base_ds()
    ds.set_active_range(2, 12)
    return ds


@pytest.fixture()
def ds_trimmed():
    ds = create_base_ds()
    ds.set_active_range(2, 12)
    return ds.trim()


@pytest.fixture()
def df_trimmed(ds_trimmed):
    return ds_trimmed


@pytest.fixture()
def df_concat(ds_trimmed):
    df = ds_trimmed
    df1 = df[:2]   # length 2
    df2 = df[2:3]  # length 1
    df3 = df[3:7]  # length 4
    df4 = df[7:10]  # length 3
    return vaex.concat([df1, df2, df3, df4])


# for when only the type of executor matters
@pytest.fixture(params=['df_trimmed', 'df_remote'])
def df_executor(request, df_trimmed, df_remote):
    named = dict(df_trimmed=df_trimmed, df_remote=df_remote)
    return named[request.param]


@pytest.fixture(params=['ds_filtered', 'ds_half', 'ds_trimmed', 'ds_remote', 'df_concat', 'df_arrow'])
def ds(request, ds_filtered, ds_half, ds_trimmed, ds_remote, df_concat, df_arrow):
    named = dict(ds_filtered=ds_filtered, ds_half=ds_half, ds_trimmed=ds_trimmed, ds_remote=ds_remote, df_concat=df_concat, df_arrow=df_arrow)
    return named[request.param]


@pytest.fixture
def df(ds):
    return ds


@pytest.fixture(params=['ds_filtered', 'ds_half', 'ds_trimmed', 'df_concat', 'df_arrow'])
def ds_local(request, ds_filtered, ds_half, ds_trimmed, df_concat, df_arrow):
    named = dict(ds_filtered=ds_filtered, ds_half=ds_half, ds_trimmed=ds_trimmed, df_concat=df_concat, df_arrow=df_arrow)
    return named[request.param]


# in some cases it is not worth testing with the arrow version
@pytest.fixture(params=['ds_filtered', 'ds_half', 'ds_trimmed', 'df_concat'])
def df_local_non_arrow(request, ds_filtered, ds_half, ds_trimmed, df_concat):
    named = dict(ds_filtered=ds_filtered, ds_half=ds_half, ds_trimmed=ds_trimmed, df_concat=df_concat)
    return named[request.param]


@pytest.fixture
def df_local(ds_local):
    return ds_local


@pytest.fixture
def df_arrow(df_arrow_raw):
    # we add the filter and virtual columns again to avoid the expression rewriting
    df = df_arrow_raw.as_numpy().drop_filter()
    del df['z']
    df.select('(x >= 0) & (x < 10)', name=vaex.dataset.FILTER_SELECTION_NAME)
    df.add_virtual_column("z", "x+t*y")
    return df


@pytest.fixture
def df_arrow_raw(df_filtered):
    df = df_filtered.copy()
    df.drop('obj', inplace=True)
    df.drop('timedelta', inplace=True)
    df.columns = {k: vaex.array_types.to_arrow(v, convert_to_native=True) for k, v in df.columns.items()}
    return df


@pytest.fixture(params=[ds_half, ds_trimmed], ids=['ds_half', 'ds_trimmed'])
def ds_no_filter(request):
    return request.param()

def create_filtered():
    ds = create_base_ds()
    ds.select('(x >= 0) & (x < 10)', name=vaex.dataset.FILTER_SELECTION_NAME)
    return ds

def create_base_ds():
    dataset = vaex.dataset.DatasetArrays("dataset")
    x = np.arange(-2, 40, dtype=">f8").reshape((-1,21)).T.copy()[:,0]
    y = y = x ** 2
    ints = np.arange(-2,19, dtype="i8")
    ints[0] = 2**62+1
    ints[1] = -2**62+1
    ints[2] = -2**62-1
    ints[0+10] = 2**62+1
    ints[1+10] = -2**62+1
    ints[2+10] = -2**62-1
    dataset.add_column("x", x)
    dataset.add_column("y", y)
    # m = x.copy()
    m = np.arange(-2, 40, dtype=">f8").reshape((-1,21)).T.copy()[:,0]
    ma_value = 77777
    m[-1+10+2] = ma_value
    m[-1+20] = ma_value
    m = np.ma.array(m, mask=m==ma_value)

    n = x.copy()
    n[-2+10] = np.nan
    n[-2+20] = np.nan

    nm = x.copy()
    nm[-2+10] = np.nan
    nm[-2+20] = np.nan
    nm[-1+10] = ma_value
    nm[-1+20] = ma_value
    nm = np.ma.array(nm, mask=nm==ma_value)

    mi = np.ma.array(m.data.astype(np.int64), mask=m.data==ma_value, fill_value=88888)
    dataset.add_column("m", m)
    dataset.add_column('n', n)
    dataset.add_column('nm', nm)
    dataset.add_column("mi", mi)
    dataset.add_column("ints", ints)

    name = np.array(list(map(lambda x: str(x) + "bla" + ('_' * int(x)), x)), dtype='U') #, dtype=np.string_)
    dataset.add_column("name", np.array(name))
    dataset.add_column("name_arrow", vaex.string_column(name))

    obj_data = np.array(['train', 'false' , True, 1, 30., np.nan, 'something', 'something a bit longer resembling a sentence?!', -10000, 'this should be masked'], dtype='object')
    obj_mask = np.array([False] * 9 + [True])
    obj = nm.copy().astype('object')
    obj[2:12] = np.ma.MaskedArray(data=obj_data, mask=obj_mask, dtype='object')
    dataset.add_column("obj", obj, dtype=np.dtype('O'))

    booleans = np.ones(21, dtype=np.bool)
    booleans[[4, 6, 8, 14, 16, 19]] = False
    dataset.add_column("bool", booleans)

    datetime = np.array(['2016-02-29T22:02:02.32', '2013-01-17T01:02:03.32', '2017-11-11T08:15:15.00',
                         '1995-04-01T05:55:55.55', '2000-01-01T00:00:00.00', '2019-03-05T09:12:13.51',
                         '1993-10-15T17:23:47.00', '2001-09-15T00:00:00.15', '2019-02-18T13:12:10.09',
                         '1991-07-12T16:17:33.11', '2005-05-05T05:05:05.05', '2011-08-27T03:06:15.00',
                         '1999-07-09T09:01:33.21', '2018-04-04T17:30:00.00', '2012-12-01T21:00:00.01',
                         '1994-05-02T11:22:33.00', '2003-07-02T22:33:00.00', '2014-06-03T06:30:00.00',
                         '1997-09-04T20:31:00.11', '2004-02-24T04:00:00.00', '2000-06-15T12:30:30.00',
                         ],dtype=np.datetime64)
    timedelta = datetime - np.datetime64('1996-05-17T16:45:00.00')
    dataset.add_column("datetime", datetime)
    dataset.add_column("timedelta", timedelta)
    dataset.add_column("123456", x)  # a column that will have an alias

    dataset.add_virtual_column("z", "x+t*y")
    dataset.set_variable("t", 1.)

    return dataset._readonly()

"""
Vaex is a library for dealing with larger than memory DataFrames (out of core).

The most important class (datastructure) in vaex is the :class:`.DataFrame`. A DataFrame is obtained by either opening
the example dataset:

>>> import vaex
>>> df = vaex.example()

Or using :func:`open` to open a file.

>>> df1 = vaex.open("somedata.hdf5")
>>> df2 = vaex.open("somedata.fits")
>>> df2 = vaex.open("somedata.arrow")
>>> df4 = vaex.open("somedata.csv")

Or connecting to a remove server:

>>> df_remote = vaex.open("http://try.vaex.io/nyc_taxi_2015")


A few strong features of vaex are:

 * Performance: works with huge tabular data, process over a billion (> 10\\ :sup:`9`\\ ) rows/second.
 * Expression system / Virtual columns: compute on the fly, without wasting ram.
 * Memory efficient: no memory copies when doing filtering/selections/subsets.
 * Visualization: directly supported, a one-liner is often enough.
 * User friendly API: you will only need to deal with a DataFrame object, and tab completion + docstring will help you out: `ds.mean<tab>`, feels very similar to Pandas.
 * Very fast statistics on N dimensional grids such as histograms, running mean, heatmaps.


Follow the tutorial at https://docs.vaex.io/en/latest/tutorial.html to learn how to use vaex.

"""  # -*- coding: utf-8 -*-
from __future__ import print_function
import glob
import re
import six

import vaex.dataframe
import vaex.dataset
from vaex.functions import register_function
from . import stat
# import vaex.file
# import vaex.export
from .delayed import delayed
from .groupby import *
from . import agg
import vaex.datasets
# import vaex.plot
# from vaex.dataframe import DataFrame
# del ServerRest, DataFrame

import vaex.settings
import logging
import pkg_resources
import os
from functools import reduce

try:
    from . import version
except:
    import sys
    print("version file not found, please run git/hooks/post-commit or git/hooks/post-checkout and/or install them as hooks (see git/README)", file=sys.stderr)
    raise

__version__ = version.get_versions()


def app(*args, **kwargs):
    """Create a vaex app, the QApplication mainloop must be started.

    In ipython notebook/jupyter do the following:

    >>> import vaex.ui.main # this causes the qt api level to be set properly
    >>> import vaex

    Next cell:

    >>> %gui qt

    Next cell:

    >>> app = vaex.app()

    From now on, you can run the app along with jupyter

    """

    import vaex.ui.main
    return vaex.ui.main.VaexApp()


def _convert_name(filenames, shuffle=False, suffix=None):
    '''Convert a filename (or list of) to a filename with .hdf5 and optionally a -shuffle or other suffix'''
    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    base = filenames[0]
    if shuffle:
        base += '-shuffle'
    if suffix:
        base += suffix
    if len(filenames) > 1:
        return base + "_and_{}_more.hdf5".format(len(filenames)-1)
    else:
        return base + ".hdf5"


def open(path, convert=False, shuffle=False, copy_index=False, *args, **kwargs):
    """Open a DataFrame from file given by path.

    Example:

    >>> df = vaex.open('sometable.hdf5')
    >>> df = vaex.open('somedata*.csv', convert='bigdata.hdf5')

    :param str or list path: local or absolute path to file, or glob string, or list of paths
    :param convert: convert files to an hdf5 file for optimization, can also be a path
    :param bool shuffle: shuffle converted DataFrame or not
    :param args: extra arguments for file readers that need it
    :param kwargs: extra keyword arguments
    :param bool copy_index: copy index when source is read via pandas
    :return: return a DataFrame on success, otherwise None
    :rtype: DataFrame

    S3 support:

    Vaex supports streaming in hdf5 files from Amazon AWS object storage S3.
    Files are by default cached in $HOME/.vaex/file-cache/s3 such that successive access
    is as fast as native disk access. The following url parameters control S3 options:

     * anon: Use anonymous access or not (false by default). (Allowed values are: true,True,1,false,False,0)
     * use_cache: Use the disk cache or not, only set to false if the data should be accessed once. (Allowed values are: true,True,1,false,False,0)
     * profile_name and other arguments are passed to :py:class:`s3fs.core.S3FileSystem`

    All arguments can also be passed as kwargs, but then arguments such as `anon` can only be a boolean, not a string.

    Examples:

    >>> df = vaex.open('s3://vaex/taxi/yellow_taxi_2015_f32s.hdf5?anon=true')
    >>> df = vaex.open('s3://vaex/taxi/yellow_taxi_2015_f32s.hdf5', anon=True)  # Note that anon is a boolean, not the string 'true'
    >>> df = vaex.open('s3://mybucket/path/to/file.hdf5?profile_name=myprofile')

    """
    import vaex
    try:
        if path in aliases:
            path = aliases[path]
        if path.startswith("http://") or path.startswith("ws://") or \
           path.startswith("vaex+http://") or path.startswith("vaex+ws://"):  # TODO: think about https and wss
            server, name = path.rsplit("/", 1)
            url = urlparse(path)
            if '?' in name:
                name = name[:name.index('?')]
            extra_args = {key: values[0] for key, values in parse_qs(url.query).items()}
            if 'token' in extra_args:
                kwargs['token'] = extra_args['token']
            if 'token_trusted' in extra_args:
                kwargs['token_trusted'] = extra_args['token_trusted']
            client = vaex.connect(server, **kwargs)
            return client[name]
        if path.startswith("cluster"):
            import vaex.distributed
            return vaex.distributed.open(path, *args, **kwargs)
        else:
            import vaex.file
            import glob
            if isinstance(path, str):
                paths = [path]
            else:
                paths = path
            filenames = []
            for path in paths:
                # TODO: can we do glob with s3?
                if path.startswith('s3://'):
                    filenames.append(path)
                else:
                    # sort to get predictable behaviour (useful for testing)
                    filenames.extend(list(sorted(glob.glob(path))))
            ds = None
            if len(filenames) == 0:
                raise IOError('Could not open file: {}, it does not exist'.format(path))
            filename_hdf5 = _convert_name(filenames, shuffle=shuffle)
            filename_hdf5_noshuffle = _convert_name(filenames, shuffle=False)
            if len(filenames) == 1:
                path = filenames[0]
                naked_path = path
                if '?' in naked_path:
                    naked_path = naked_path[:naked_path.index('?')]
                ext = os.path.splitext(naked_path)[1]
                if os.path.exists(filename_hdf5) and convert:  # also check mtime?
                    ds = vaex.file.open(filename_hdf5)
                else:
                    if ext == '.csv' or naked_path.endswith(".csv.bz2"):  # special support for csv.. should probably approach it a different way
                        csv_convert = filename_hdf5 if convert else False
                        ds = from_csv(path, copy_index=copy_index, convert=csv_convert, **kwargs)
                    else:
                        ds = vaex.file.open(path, *args, **kwargs)
                        if convert and ds:
                            ds.export_hdf5(filename_hdf5, shuffle=shuffle)
                            ds = vaex.file.open(filename_hdf5)  # argument were meant for pandas?
                if ds is None:
                    if os.path.exists(path):
                        raise IOError('Could not open file: {}, did you install vaex-hdf5? Is the format supported?'.format(path))
                    if os.path.exists(path):
                        raise IOError('Could not open file: {}, it does not exist?'.format(path))
            elif len(filenames) > 1:
                if convert not in [True, False]:
                    filename_hdf5 = convert
                else:
                    filename_hdf5 = _convert_name(filenames, shuffle=shuffle)
                if os.path.exists(filename_hdf5) and convert:  # also check mtime
                    ds = open(filename_hdf5)
                else:
                    # with ProcessPoolExecutor() as executor:
                    # executor.submit(read_csv_and_convert, filenames, shuffle=shuffle, **kwargs)
                    DataFrames = []
                    for filename in filenames:
                        DataFrames.append(open(filename, convert=bool(convert), shuffle=shuffle, **kwargs))
                    ds = vaex.dataframe.DataFrameConcatenated(DataFrames)
                    if convert:
                        ds.export_hdf5(filename_hdf5, shuffle=shuffle)
                        ds = vaex.file.open(filename_hdf5)

        if ds is None:
            raise IOError('Unknown error opening: {}'.format(path))
        return ds
    except:
        logging.getLogger("vaex").error("error opening %r" % path)
        raise


def open_many(filenames):
    """Open a list of filenames, and return a DataFrame with all DataFrames concatenated.

    :param list[str] filenames: list of filenames/paths
    :rtype: DataFrame
    """
    dfs = []
    for filename in filenames:
        filename = filename.strip()
        if filename and filename[0] != "#":
            dfs.append(open(filename))
    return vaex.dataframe.DataFrameConcatenated(dfs=dfs)


def from_samp(username=None, password=None):
    """Connect to a SAMP Hub and wait for a single table load event, disconnect, download the table and return the DataFrame.

    Useful if you want to send a single table from say TOPCAT to vaex in a python console or notebook.
    """
    print("Waiting for SAMP message...")
    import vaex.samp
    t = vaex.samp.single_table(username=username, password=password)
    return from_astropy_table(t.to_table())


def from_astropy_table(table):
    """Create a vaex DataFrame from an Astropy Table."""
    import vaex.file.other
    return vaex.file.other.DatasetAstropyTable(table=table)


def from_dict(data):
    """Create an in memory dataset from a dict with column names as keys and list/numpy-arrays as values

    Example

    >>> data = {'A':[1,2,3],'B':['a','b','c']}
    >>> vaex.from_dict(data)
      #    A    B
      0    1   'a'
      1    2   'b'
      2    3   'c'

    :param data: A dict of {column:[value, value,...]}
    :rtype: DataFrame

    """
    return vaex.from_arrays(**data)


def from_items(*items):
    """Create an in memory DataFrame from numpy arrays, in contrast to from_arrays this keeps the order of columns intact (for Python < 3.6).

    Example

    >>> import vaex, numpy as np
    >>> x = np.arange(5)
    >>> y = x ** 2
    >>> vaex.from_items(('x', x), ('y', y))
      #    x    y
      0    0    0
      1    1    1
      2    2    4
      3    3    9
      4    4   16

    :param items: list of [(name, numpy array), ...]
    :rtype: DataFrame

    """
    import numpy as np
    df = vaex.dataframe.DataFrameArrays("array")
    for name, array in items:
        df.add_column(name, np.asanyarray(array))
    return df


def from_arrays(**arrays):
    """Create an in memory DataFrame from numpy arrays.

    Example

    >>> import vaex, numpy as np
    >>> x = np.arange(5)
    >>> y = x ** 2
    >>> vaex.from_arrays(x=x, y=y)
      #    x    y
      0    0    0
      1    1    1
      2    2    4
      3    3    9
      4    4   16
    >>> some_dict = {'x': x, 'y': y}
    >>> vaex.from_arrays(**some_dict)  # in case you have your columns in a dict
      #    x    y
      0    0    0
      1    1    1
      2    2    4
      3    3    9
      4    4   16

    :param arrays: keyword arguments with arrays
    :rtype: DataFrame
    """
    import numpy as np
    import six
    from .column import Column
    df = vaex.dataframe.DataFrameArrays("array")
    for name, array in arrays.items():
        if isinstance(array, Column):
            df.add_column(name, array)
        else:
            array = np.asanyarray(array)
            df.add_column(name, array)
    return df

def from_arrow_table(table):
    """Creates a vaex DataFrame from an arrow Table.

    :rtype: DataFrame
    """
    from vaex_arrow.convert import vaex_df_from_arrow_table
    return vaex_df_from_arrow_table(table=table)

def from_scalars(**kwargs):
    """Similar to from_arrays, but convenient for a DataFrame of length 1.

    Example:

    >>> import vaex
    >>> df = vaex.from_scalars(x=1, y=2)

    :rtype: DataFrame
    """
    import numpy as np
    return from_arrays(**{k: np.array([v]) for k, v in kwargs.items()})


def from_pandas(df, name="pandas", copy_index=False, index_name="index"):
    """Create an in memory DataFrame from a pandas DataFrame.

    :param: pandas.DataFrame df: Pandas DataFrame
    :param: name: unique for the DataFrame

    >>> import vaex, pandas as pd
    >>> df_pandas = pd.from_csv('test.csv')
    >>> df = vaex.from_pandas(df_pandas)

    :rtype: DataFrame
    """
    import six
    import pandas as pd
    import numpy as np
    vaex_df = vaex.dataframe.DataFrameArrays(name)

    def add(name, column):
        values = column.values
        # the first test is to support (partially) pandas 0.23
        if hasattr(pd.core.arrays, 'integer') and isinstance(values, pd.core.arrays.integer.IntegerArray):
            values = np.ma.array(values._data, mask=values._mask)
        try:
            vaex_df.add_column(name, values)
        except Exception as e:
            print("could not convert column %s, error: %r, will try to convert it to string" % (name, e))
            try:
                values = values.astype("S")
                vaex_df.add_column(name, values)
            except Exception as e:
                print("Giving up column %s, error: %r" % (name, e))
    for name in df.columns:
        add(name, df[name])
    if copy_index:
        add(index_name, df.index)
    return vaex_df


def from_ascii(path, seperator=None, names=True, skip_lines=0, skip_after=0, **kwargs):
    """
    Create an in memory DataFrame from an ascii file (whitespace seperated by default).

    >>> ds = vx.from_ascii("table.asc")
    >>> ds = vx.from_ascii("table.csv", seperator=",", names=["x", "y", "z"])

    :param path: file path
    :param seperator: value seperator, by default whitespace, use "," for comma seperated values.
    :param names: If True, the first line is used for the column names, otherwise provide a list of strings with names
    :param skip_lines: skip lines at the start of the file
    :param skip_after: skip lines at the end of the file
    :param kwargs:
    :rtype: DataFrame
    """

    import vaex.ext.readcol as rc
    ds = vaex.dataframe.DataFrameArrays(path)
    if names not in [True, False]:
        namelist = names
        names = False
    else:
        namelist = None
    data = rc.readcol(path, fsep=seperator, asdict=namelist is None, names=names, skipline=skip_lines, skipafter=skip_after, **kwargs)
    if namelist:
        for name, array in zip(namelist, data.T):
            ds.add_column(name, array)
    else:
        for name, array in data.items():
            ds.add_column(name, array)
    return ds


def from_json(path_or_buffer, orient=None, precise_float=False, lines=False, copy_index=False, **kwargs):
    """ A method to read a JSON file using pandas, and convert to a DataFrame directly.

    :param str path_or_buffer: a valid JSON string or file-like, default: None
    The string could be a URL. Valid URL schemes include http, ftp, s3,
    gcs, and file. For file URLs, a host is expected. For instance, a local
    file could be ``file://localhost/path/to/table.json``
    :param str orient: Indication of expected JSON string format. Allowed values are
    ``split``, ``records``, ``index``, ``columns``, and ``values``.
    :param bool precise_float: Set to enable usage of higher precision (strtod) function when
    decoding string to double values. Default (False) is to use fast but less precise builtin functionality
    :param bool lines: Read the file as a json object per line.

    :rtype: DataFrame
    """
    # Check for unsupported kwargs
    if kwargs.get('typ') == 'series':
        raise ValueError('`typ` must be set to `"frame"`.')
    if kwargs.get('numpy') == True:
        raise ValueError('`numpy` must be set to `False`.')
    if kwargs.get('chunksize') is not None:
        raise ValueError('`chunksize` must be `None`.')

    import pandas as pd
    return from_pandas(pd.read_json(path_or_buffer, orient=orient, precise_float=precise_float, lines=lines, **kwargs),
                       copy_index=copy_index)


def from_csv(filename_or_buffer, copy_index=False, chunk_size=None, convert=False, **kwargs):
    """
    Read a CSV file as a DataFrame, and optionally convert to an hdf5 file.

    :param str or file filename_or_buffer: CSV file path or file-like
    :param bool copy_index: copy index when source is read via Pandas
    :param int chunk_size: if the CSV file is too big to fit in the memory this parameter can be used to read
        CSV file in chunks. For example:

        >>> import vaex
        >>> for i, df in enumerate(vaex.from_csv('taxi.csv', chunk_size=100_000)):
        >>>     df = df[df.passenger_count < 6]
        >>>     df.export_hdf5(f'taxi_{i:02}.hdf5')

    :param bool or str convert: convert files to an hdf5 file for optimization, can also be a path. The CSV
        file will be read in chunks: either using the provided chunk_size argument, or a default size. Each chunk will
        be saved as a separate hdf5 file, then all of them will be combined into one hdf5 file. So for a big CSV file
        you will need at least double of extra space on the disk. Default chunk_size for converting is 5 million rows,
        which corresponds to around 1Gb memory on an example of NYC Taxi dataset.
    :param kwargs: extra keyword arguments, currently passed to Pandas read_csv function, but the implementation might
        change in future versions.
    :returns: DataFrame
    """
    if not convert:
        return _from_csv_read(filename_or_buffer=filename_or_buffer, copy_index=copy_index,
                              chunk_size=chunk_size, **kwargs)
    else:
        if chunk_size is None:
            # make it memory efficient by default
            chunk_size = 5_000_000
        return _from_csv_convert_and_read(filename_or_buffer=filename_or_buffer, copy_index=copy_index,
                                          maybe_convert_path=convert, chunk_size=chunk_size, **kwargs)


def _from_csv_read(filename_or_buffer, copy_index, chunk_size, **kwargs):
    import pandas as pd
    if not chunk_size:
        full_df = pd.read_csv(filename_or_buffer, **kwargs)
        return from_pandas(full_df, copy_index=copy_index)
    else:
        def iterator():
            chunk_iterator = pd.read_csv(filename_or_buffer, chunksize=chunk_size, **kwargs)
            for chunk_df in chunk_iterator:
                yield from_pandas(chunk_df, copy_index=copy_index)
        return iterator()


def _from_csv_convert_and_read(filename_or_buffer, copy_index, maybe_convert_path, chunk_size, **kwargs):
    # figure out the CSV file path
    if isinstance(filename_or_buffer, str):
        csv_path = filename_or_buffer
    elif isinstance(maybe_convert_path, str):
        csv_path = re.sub(r'\.hdf5$', '', str(maybe_convert_path), flags=re.IGNORECASE)
    else:
        raise ValueError('Cannot derive filename to use for converted HDF5 file, '
                         'please specify it using convert="my.csv.hdf5"')

    # reuse a previously converted HDF5 file
    import vaex.file
    combined_hdf5 = _convert_name(csv_path)
    if os.path.exists(combined_hdf5):
        return vaex.file.open(combined_hdf5)

    # convert CSV chunks to separate HDF5 files
    import pandas as pd
    converted_paths = []
    csv_reader = pd.read_csv(filename_or_buffer, chunksize=chunk_size, **kwargs)
    for i, df_pandas in enumerate(csv_reader):
        df = from_pandas(df_pandas, copy_index=copy_index)
        filename_hdf5 = _convert_name(csv_path, suffix='_chunk%d' % i)
        df.export_hdf5(filename_hdf5, shuffle=False)
        converted_paths.append(filename_hdf5)
        logger.info('saved chunk #%d to %s' % (i, filename_hdf5))

    # combine chunks into one HDF5 file
    if len(converted_paths) == 1:
        # no need to merge several HDF5 files
        os.rename(converted_paths[0], combined_hdf5)
    else:
        logger.info('converting %d chunks into single HDF5 file %s' % (len(converted_paths), combined_hdf5))
        dfs = [vaex.file.open(p) for p in converted_paths]
        df_combined = vaex.dataframe.DataFrameConcatenated(dfs)
        df_combined.export_hdf5(combined_hdf5, shuffle=False)

        logger.info('deleting %d chunk files' % len(converted_paths))
        for df, df_path in zip(dfs, converted_paths):
            try:
                df.close_files()
                os.remove(df_path)
            except Exception as e:
                logger.error('Could not close or delete intermediate hdf5 file %s used to convert %s to hdf5: %s' % (
                    df_path, csv_path, e))

    return vaex.file.open(combined_hdf5)


def read_csv(filepath_or_buffer, **kwargs):
    '''Alias to from_csv.'''
    return from_csv(filepath_or_buffer, **kwargs)


def read_csv_and_convert(path, shuffle=False, copy_index=False, **kwargs):
    '''Convert a path (or glob pattern) to a single hdf5 file, will open the hdf5 file if exists.

    Example:
            >>> vaex.read_csv_and_convert('test-*.csv', shuffle=True)  # this may take a while
            >>> vaex.read_csv_and_convert('test-*.csv', shuffle=True)  # 2nd time it is instant

    :param str path: path of file or glob pattern for multiple files
    :param bool shuffle: shuffle DataFrame when converting to hdf5
    :param bool copy_index: by default pandas will create an index (row number), set to true if you want to include this as a column.
    :param kwargs: parameters passed to pandas' read_cvs

    '''
    from concurrent.futures import ProcessPoolExecutor
    import pandas as pd
    filenames = glob.glob(path)
    if len(filenames) > 1:
        filename_hdf5 = _convert_name(filenames, shuffle=shuffle)
        filename_hdf5_noshuffle = _convert_name(filenames, shuffle=False)
        if not os.path.exists(filename_hdf5):
            if not os.path.exists(filename_hdf5_noshuffle):
                # with ProcessPoolExecutor() as executor:
                # executor.submit(read_csv_and_convert, filenames, shuffle=shuffle, **kwargs)
                for filename in filenames:
                    read_csv_and_convert(filename, shuffle=shuffle, copy_index=copy_index, **kwargs)
                ds = open_many([_convert_name(k, shuffle=shuffle) for k in filenames])
            else:
                ds = open(filename_hdf5_noshuffle)
            ds.export_hdf5(filename_hdf5, shuffle=shuffle)
        return open(filename_hdf5)
    else:
        filename = filenames[0]
        filename_hdf5 = _convert_name(filename, shuffle=shuffle)
        filename_hdf5_noshuffle = _convert_name(filename, shuffle=False)
        if not os.path.exists(filename_hdf5):
            if not os.path.exists(filename_hdf5_noshuffle):
                df = pd.read_csv(filename, **kwargs)
                ds = from_pandas(df, copy_index=copy_index)
            else:
                ds = open(filename_hdf5_noshuffle)
            ds.export_hdf5(filename_hdf5, shuffle=shuffle)
        return open(filename_hdf5)


aliases = vaex.settings.main.auto_store_dict("aliases")

# py2/p3 compatibility
try:
    from urllib.parse import urlparse, parse_qs
except ImportError:
    from urlparse import urlparse, parse_qs


def connect(url, **kwargs):
    """Connect to hostname supporting the vaex web api.

    :param str hostname: hostname or ip address of server
    :rtype: vaex.server.client.Client
    """
    # dispatch to vaex.server package
    from vaex.server import connect
    return connect(url, **kwargs)


def example():
    """Returns an example DataFrame which comes with vaex for testing/learning purposes.

    :rtype: DataFrame
    """
    return vaex.datasets.helmi_de_zeeuw_10percent.fetch()


def zeldovich(dim=2, N=256, n=-2.5, t=None, scale=1, seed=None):
    """Creates a zeldovich DataFrame.
    """
    import vaex.file
    return vaex.file.other.Zeldovich(dim=dim, N=N, n=n, t=t, scale=scale)


def set_log_level_debug():
    """set log level to debug"""
    import logging
    logging.getLogger("vaex").setLevel(logging.DEBUG)


def set_log_level_info():
    """set log level to info"""
    import logging
    logging.getLogger("vaex").setLevel(logging.INFO)


def set_log_level_warning():
    """set log level to warning"""
    import logging
    logging.getLogger("vaex").setLevel(logging.WARNING)


def set_log_level_exception():
    """set log level to exception"""
    import logging
    logging.getLogger("vaex").setLevel(logging.FATAL)


def set_log_level_off():
    """Disabled logging"""
    import logging
    logging.disable(logging.CRITICAL)


format = "%(levelname)s:%(threadName)s:%(name)s:%(message)s"
logging.basicConfig(level=logging.INFO, format=format)
DEBUG_MODE = bool(os.environ.get('VAEX_DEBUG', ''))
if DEBUG_MODE:
    logging.basicConfig(level=logging.DEBUG)
    set_log_level_debug()
else:
    # logging.basicConfig(level=logging.DEBUG)
    set_log_level_warning()

import_script = os.path.expanduser("~/.vaex/vaex_import.py")
if os.path.exists(import_script):
    try:
        with open(import_script) as f:
            code = compile(f.read(), import_script, 'exec')
            exec(code)
    except:
        import traceback
        traceback.print_stack()


logger = logging.getLogger('vaex')


def register_dataframe_accessor(name, cls=None, override=False):
    """Registers a new accessor for a dataframe

    See vaex.geo for an example.
    """
    def wrapper(cls):
        old_value = getattr(vaex.dataframe.DataFrame, name, None)
        if old_value is not None and override is False:
            raise ValueError("DataFrame already has a property/accessor named %r (%r)" % (name, old_value) )

        def get_accessor(self):
            if name in self.__dict__:
                return self.__dict__[name]
            else:
                self.__dict__[name] = cls(self)
            return self.__dict__[name]
        setattr(vaex.dataframe.DataFrame, name, property(get_accessor))
        return cls
    if cls is None:
        return wrapper
    else:
        return wrapper(cls)


for entry in pkg_resources.iter_entry_points(group='vaex.namespace'):
    logger.warning('(DEPRECATED, use vaex.dataframe.accessor) adding vaex namespace: ' + entry.name)
    try:
        add_namespace = entry.load()
        add_namespace()
    except Exception:
        logger.exception('issue loading ' + entry.name)

_df_lazy_accessors = {}


class _lazy_accessor(object):
    def __init__(self, name, scope, loader):
        """When adding an accessor geo.cone, scope=='geo', name='cone', scope may be falsy"""
        self.loader = loader
        self.name = name
        self.scope = scope

    def __call__(self, obj):
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]
        else:
            cls = self.loader()
            accessor = cls(obj)
            obj.__dict__[self.name] = accessor
            fullname = self.name
            if self.scope:
                fullname = self.scope + '.' + self.name
            if fullname in _df_lazy_accessors:
                for name, scope, loader in _df_lazy_accessors[fullname]:
                    assert fullname == scope
                    setattr(cls, name, property(_lazy_accessor(name, scope, loader)))
        return obj.__dict__[self.name]


def _add_lazy_accessor(name, loader, target_class=vaex.dataframe.DataFrame):
    """Internal use see tests/internal/accessor_test.py for usage

    This enables us to have df.foo.bar accessors that lazily loads the modules.
    """
    parts = name.split('.')
    target_class = vaex.dataframe.DataFrame
    if len(parts) == 1:
        setattr(target_class, parts[0], property(_lazy_accessor(name, None, loader)))
    else:
        scope = ".".join(parts[:-1])
        if scope not in _df_lazy_accessors:
            _df_lazy_accessors[scope] = []
        _df_lazy_accessors[scope].append((parts[-1], scope, loader))


for entry in pkg_resources.iter_entry_points(group='vaex.dataframe.accessor'):
    logger.debug('adding vaex accessor: ' + entry.name)
    def loader(entry=entry):
        return entry.load()
    _add_lazy_accessor(entry.name, loader)


for entry in pkg_resources.iter_entry_points(group='vaex.plugin'):
    logger.debug('adding vaex plugin: ' + entry.name)
    try:
        add_namespace = entry.load()
        add_namespace()
    except Exception:
        logger.exception('issue loading ' + entry.name)


def concat(dfs):
    '''Concatenate a list of DataFrames.

    :rtype: DataFrame
    '''
    ds = reduce((lambda x, y: x.concat(y)), dfs)
    return ds

def vrange(start, stop, step=1, dtype='f8'):
    """Creates a virtual column which is the equivalent of numpy.arange, but uses 0 memory"""
    from .column import ColumnVirtualRange
    return ColumnVirtualRange(start, stop, step, dtype)

def string_column(strings):
    from vaex_arrow.convert import column_from_arrow_array
    import pyarrow as pa
    return column_from_arrow_array(pa.array(strings))

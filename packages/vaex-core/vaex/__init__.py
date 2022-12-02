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

- Performance: works with huge tabular data, process over a billion (> 10\\ :sup:`9`\\ ) rows/second.
- Expression system / Virtual columns: compute on the fly, without wasting ram.
- Memory efficient: no memory copies when doing filtering/selections/subsets.
- Visualization: directly supported, a one-liner is often enough.
- User friendly API: you will only need to deal with a DataFrame object, and tab completion + docstring will help you out: `ds.mean<tab>`, feels very similar to Pandas.
- Very fast statistics on N dimensional grids such as histograms, running mean, heatmaps.


Follow the tutorial at https://docs.vaex.io/en/latest/tutorial.html to learn how to use vaex.

"""  # -*- coding: utf-8 -*-
import logging as root_logging
import os
from typing import Dict, List
from urllib.parse import urlparse, parse_qs

# first configure logging, which also imports vaex.settings
import vaex.logging
# import this to be explicit
import vaex.settings

import vaex.dataframe
import vaex.dataset
from vaex.docstrings import docsubst
from vaex.registry import register_function
from vaex import functions, struct
from . import stat
# import vaex.file
# import vaex.export
from .delayed import delayed
from .groupby import *
from . import agg
import vaex.datasets

# Re-export these so users can type hint with eg vaex.DataFrame
from vaex.dataframe import DataFrame as DataFrame
from vaex.expression import Expression as Expression

import vaex.progress

try:
    from sys import version_info
    if version_info[:2] >= (3, 10):
        from importlib.metadata import entry_points
    else:
        from importlib_metadata import entry_points, __version__ as importlib_metadata_version
        if int(importlib_metadata_version.split(".")[0]) < 4.0:
            raise ImportError("vaex requires importlib_metadata >= 4.0 when installed")
except ImportError:
    import pkg_resources
    entry_points = pkg_resources.iter_entry_points

try:
    from . import version
except:
    import sys
    print("version file not found, please run git/hooks/post-commit or git/hooks/post-checkout and/or install them as hooks (see git/README)", file=sys.stderr)
    raise

logger = root_logging.getLogger('vaex')
DEBUG_MODE = os.environ.get('VAEX_DEBUG', '')
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


@docsubst
def open(path, convert=False, progress=None, shuffle=False, fs_options={}, fs=None, *args, **kwargs):
    """Open a DataFrame from file given by path.

    Example:

    >>> df = vaex.open('sometable.hdf5')
    >>> df = vaex.open('somedata*.csv', convert='bigdata.hdf5')

    :param str or list path: local or absolute path to file, or glob string, or list of paths
    :param convert: Uses `dataframe.export` when convert is a path. If True, ``convert=path+'.hdf5'``
                    The conversion is skipped if the input file or conversion argument did not change.
    :param progress: (*Only applies when convert is not False*) {progress}
    :param bool shuffle: shuffle converted DataFrame or not
    :param dict fs_options: Extra arguments passed to an optional file system if needed. See below
    :param group: (optional) Specify the group to be read from and HDF5 file. By default this is set to "/table".
    :param fs: Apache Arrow FileSystem object, or FSSpec FileSystem object, if specified, fs_options should be empty.
    :param args: extra arguments for file readers that need it
    :param kwargs: extra keyword arguments
    :return: return a DataFrame on success, otherwise None
    :rtype: DataFrame

    Note: From version 4.14.0 `vaex.open()` will lazily read CSV files.
    If you prefer to read the entire CSV file into memory, use `vaex.from_csv()` or `vaex.from_csv_arrow()` instead.

    Cloud storage support:

    Vaex supports streaming of HDF5 files from Amazon AWS S3 and Google Cloud Storage.
    Files are by default cached in $HOME/.vaex/file-cache/(s3|gs) such that successive access
    is as fast as native disk access.

    Amazon AWS S3 options:

    The following common fs_options are used for S3 access:

     * `anon`: Use anonymous access or not (false by default). (Allowed values are: true,True,1,false,False,0)
     * `anonymous` - Alias for `anon`
     * `cache`: Use the disk cache or not, only set to false if the data should be accessed once. (Allowed values are: true,True,1,false,False,0)
     * `access_key` - AWS access key, if not provided will use the standard env vars, or the `~/.aws/credentials` file
     * `secret_key` - AWS secret key, similar to `access_key`
     * `profile` - If multiple profiles are present in `~/.aws/credentials`, pick this one instead of 'default', see https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html
     * `region` - AWS Region, e.g. 'us-east-1`, will be determined automatically if not provided.
     * `endpoint_override` - URL/ip to connect to, instead of AWS, e.g. 'localhost:9000' for minio

    All fs_options can also be encoded in the file path as a query string.

    Examples:

    >>> df = vaex.open('s3://vaex/taxi/yellow_taxi_2015_f32s.hdf5', fs_options={{'anonymous': True}})
    >>> df = vaex.open('s3://vaex/taxi/yellow_taxi_2015_f32s.hdf5?anon=true')
    >>> df = vaex.open('s3://mybucket/path/to/file.hdf5', fs_options={{'access_key': my_key, 'secret_key': my_secret_key}})
    >>> df = vaex.open(f's3://mybucket/path/to/file.hdf5?access_key={{my_key}}&secret_key={{my_secret_key}}')
    >>> df = vaex.open('s3://mybucket/path/to/file.hdf5?profile=myproject')

    Google Cloud Storage options:

    The following fs_options are used for GCP access:

     * token: Authentication method for GCP. Use 'anon' for annonymous access. See https://gcsfs.readthedocs.io/en/latest/index.html#credentials for more details.
     * cache: Use the disk cache or not, only set to false if the data should be accessed once. (Allowed values are: true,True,1,false,False,0).
     * project and other arguments are passed to :py:class:`gcsfs.core.GCSFileSystem`

    Examples:

    >>> df = vaex.open('gs://vaex-data/airlines/us_airline_data_1988_2019.hdf5', fs_options={{'token': None}})
    >>> df = vaex.open('gs://vaex-data/airlines/us_airline_data_1988_2019.hdf5?token=anon')
    >>> df = vaex.open('gs://vaex-data/testing/xys.hdf5?token=anon&cache=False')
    """
    import vaex
    import vaex.convert
    import vaex.csv  # need to import this to register for dask/fingerprinting
    try:
        if not isinstance(path, (list, tuple)):
            # remote and clusters only support single path, not a list
            path = vaex.file.stringyfy(path)
            if path in aliases:
                path = aliases[path]
            path = vaex.file.stringyfy(path)
            if path.startswith("http://") or path.startswith("ws://") or \
                path.startswith("vaex+wss://") or path.startswith("wss://") or \
               path.startswith("vaex+http://") or path.startswith("vaex+ws://"):
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
                import vaex.enterprise.distributed
                return vaex.enterprise.distributed.open(path, *args, **kwargs)

        import vaex.file
        import glob
        if isinstance(path, str):
            paths = [path]
        else:
            paths = path
        filenames = []
        for path in paths:
            path = vaex.file.stringyfy(path)
            if path in aliases:
                path = aliases[path]
            path = vaex.file.stringyfy(path)
            naked_path, options = vaex.file.split_options(path)
            if glob.has_magic(naked_path):
                filenames.extend(list(sorted(vaex.file.glob(path, fs_options=fs_options, fs=fs))))
            else:
                filenames.append(path)
        df = None
        if len(filenames) == 0:
            raise IOError(f'File pattern did not match anything {path}')
        filename_hdf5 = vaex.convert._convert_name(filenames, shuffle=shuffle)
        filename_hdf5_noshuffle = vaex.convert._convert_name(filenames, shuffle=False)
        if len(filenames) == 1:
            path = filenames[0]
            # # naked_path, _ = vaex.file.split_options(path, fs_options)
            _, ext, _ = vaex.file.split_ext(path)
            if convert:
                path_output = convert if isinstance(convert, str) else filename_hdf5
                vaex.convert.convert(
                    path_input=path, fs_options_input=fs_options, fs_input=fs,
                    path_output=path_output, fs_options_output=fs_options, fs_output=fs,
                    progress=progress,
                    *args, **kwargs
                )
                ds = vaex.dataset.open(path_output, fs_options=fs_options, fs=fs)
            else:
                ds = vaex.dataset.open(path, fs_options=fs_options, fs=fs, **kwargs)
            df = vaex.from_dataset(ds)
            if df is None:
                if os.path.exists(path):
                    raise IOError('Could not open file: {}, did you install vaex-hdf5? Is the format supported?'.format(path))
        elif len(filenames) > 1:
            if convert not in [True, False]:
                filename_hdf5 = convert
            else:
                filename_hdf5 = vaex.convert._convert_name(filenames, shuffle=shuffle)
            if os.path.exists(filename_hdf5) and convert:  # also check mtime
                df = vaex.open(filename_hdf5)
            else:
                dfs = []
                for filename in filenames:
                    dfs.append(vaex.open(filename, fs_options=fs_options, fs=fs, convert=bool(convert), shuffle=shuffle, **kwargs))
                df = vaex.concat(dfs)
                if convert:
                    if shuffle:
                        df = df.shuffle()
                    df.export_hdf5(filename_hdf5, progress=progress)
                    df = vaex.open(filename_hdf5)

        if df is None:
            raise IOError('Unknown error opening: {}'.format(path))
        return df
    except:
        logger.exception("error opening %r" % path)
        raise


def open_many(filenames):
    """Open a list of filenames, and return a DataFrame with all DataFrames concatenated.

    The filenames can be of any format that is supported by :py:func:`vaex.open`, namely hdf5, arrow, parquet, csv, etc.

    :param list[str] filenames: list of filenames/paths
    :rtype: DataFrame
    """
    dfs = []
    for filename in filenames:
        filename = filename.strip()
        if filename and filename[0] != "#":
            dfs.append(open(filename))
    return concat(dfs)


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
    from vaex.astro.astropy_table import DatasetAstropyTable
    ds = DatasetAstropyTable(table=table)
    return vaex.dataframe.DataFrameLocal(ds)


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
    return from_dict(dict(items))


def from_arrays(**arrays) -> vaex.dataframe.DataFrameLocal:
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
    dataset = vaex.dataset.DatasetArrays(arrays)
    return vaex.dataframe.DataFrameLocal(dataset)


def from_arrow_table(table) -> vaex.dataframe.DataFrame:
    """Creates a vaex DataFrame from an arrow Table.

    :param as_numpy: Will lazily cast columns to a NumPy ndarray.
    :rtype: DataFrame
    """
    from vaex.arrow.dataset import from_table
    return from_dataset(from_table(table=table))


def from_arrow_dataset(arrow_dataset) -> vaex.dataframe.DataFrame:
    '''Create a DataFrame from an Apache Arrow dataset.'''
    import vaex.arrow.dataset
    return from_dataset(vaex.arrow.dataset.DatasetArrow(arrow_dataset))


def from_dataset(dataset: vaex.dataset.Dataset) -> vaex.dataframe.DataFrame:
    '''Create a Vaex DataFrame from a Vaex Dataset'''
    return vaex.dataframe.DataFrameLocal(dataset)


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
    import pyarrow as pa
    columns = {}

    def add(name, column):
        values = column.values
        # the first test is to support (partially) pandas 0.23
        if hasattr(pd.core.arrays, 'integer') and isinstance(values, pd.core.arrays.integer.IntegerArray):
            values = np.ma.array(values._data, mask=values._mask)
        elif hasattr(pd.core.arrays, 'StringArray') and isinstance(values, pd.core.arrays.StringArray):
            values = pa.array(values)
        elif hasattr(pd.core.arrays, 'FloatingArray') and isinstance(values, pd.core.arrays.FloatingArray):
            values = np.ma.array(values._data, mask=values._mask)
        try:
            columns[name] = vaex.dataset.to_supported_array(values)
        except Exception as e:
            print("could not convert column %s, error: %r, will try to convert it to string" % (name, e))
            try:
                values = values.astype("S")
                columns[name] = vaex.dataset.to_supported_array(values)
            except Exception as e:
                print("Giving up column %s, error: %r" % (name, e))
    for name in df.columns:
        add(str(name), df[name])
    if copy_index:
        add(index_name, df.index)
    return from_dict(columns)


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
    ds = vaex.dataframe.DataFrameLocal()
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


@docsubst
def from_records(records : List[Dict], array_type="arrow", defaults={}) -> vaex.dataframe.DataFrame:
    '''Create a dataframe from a list of dict.

    .. warning:: This is for convenience only, for performance pass arrays to :func:`from_arrays` for instance.

    :param str array_type: {array_type}
    :param dict defaults: default values if a record has a missing entry
    '''
    arrays = dict()
    for i, record in enumerate(records):
        for name, value in record.items():
            if name not in arrays:
                # prepend None's
                arrays[name] = [defaults.get(name)] * i
            arrays[name].append(value)
        for name in arrays:
            if name not in record:
                # missing values get replaced
                arrays[name].append(defaults.get(name))
    arrays = {k: vaex.array_types.convert(v, array_type) for k, v in arrays.items()}
    return vaex.from_dict(arrays)


@docsubst
def from_csv_arrow(file, read_options=None, parse_options=None, convert_options=None, lazy=False, chunk_size="10MiB", newline_readahead="64kiB", schema_infer_fraction=0.01, fs_options={}, fs=None):
    """ Fast CSV reader using Apache Arrow. Support for lazy reading of CSV files (experimental).

    :param file: file path or file-like object
    :param read_options: PyArrow CSV read options, see https://arrow.apache.org/docs/python/generated/pyarrow.csv.ReadOptions.html
    :param parse_options: PyArrow CSV parse options, see https://arrow.apache.org/docs/python/generated/pyarrow.csv.ParseOptions.html
    :param convert_options: PyArrow CSV convert options, see https://arrow.apache.org/docs/python/generated/pyarrow.csv.ConvertOptions.html
    :param lazy: If True, the CSV file is lazily read, and the DataFrame is not stored in memory.
    :param chunk_size: The CSV is read in chunks of the specified size. Relevant only if lazy=True.
    :param newline_readahead: The size of the readahead buffer for newline detection. Relevant only if lazy=True.
    :param schema_infer_fraction: The fraction of the CSV file to read to infer the schema. Relevant only if lazy=True.
    :param fs_options: {fs_options}
    :param fs: {fs}
    :return: DataFrame
    """
    import vaex.csv
    if lazy is True:
        ds = vaex.csv.DatasetCsvLazy(file, chunk_size=chunk_size, read_options=read_options, parse_options=parse_options, convert_options=convert_options, newline_readahead=newline_readahead, schema_infer_fraction=schema_infer_fraction, fs=fs, fs_options=fs_options)
        return vaex.from_dataset(ds)
    else:
        ds = vaex.csv.DatasetCsv(file, read_options=read_options, parse_options=parse_options, convert_options=convert_options, fs=fs, fs_options=fs_options)
        return vaex.from_dataset(ds)


@docsubst
def from_csv(filename_or_buffer, copy_index=False, chunk_size=None, convert=False, fs_options={}, progress=None, fs=None, **kwargs):
    """
    Load a CSV file as a DataFrame, and optionally convert to an HDF5 file.

    :param str or file filename_or_buffer: CSV file path or file-like
    :param bool copy_index: copy index when source is read via Pandas
    :param int chunk_size: if the CSV file is too big to fit in the memory this parameter can be used to read
        CSV file in chunks. For example:

        >>> import vaex
        >>> for i, df in enumerate(vaex.read_csv('taxi.csv', chunk_size=100_000)):
        >>>     df = df[df.passenger_count < 6]
        >>>     df.export_hdf5(f'taxi_{{i:02}}.hdf5')

    :param bool or str convert: convert files to an hdf5 file for optimization, can also be a path. The CSV
        file will be read in chunks: either using the provided chunk_size argument, or a default size. Each chunk will
        be saved as a separate hdf5 file, then all of them will be combined into one hdf5 file. So for a big CSV file
        you will need at least double of extra space on the disk. Default chunk_size for converting is 5 million rows,
        which corresponds to around 1Gb memory on an example of NYC Taxi dataset.
    :param progress: (*Only applies when convert is not False*) {progress}
    :param kwargs: extra keyword arguments, currently passed to Pandas read_csv function, but the implementation might
        change in future versions.
    :returns: DataFrame
    """
    if not convert:
        return _read_csv_read(filename_or_buffer=filename_or_buffer, copy_index=copy_index,
                              fs_options=fs_options, fs=fs, chunk_size=chunk_size, **kwargs)
    else:
        if chunk_size is None:
            # make it memory efficient by default
            chunk_size = 5_000_000
        import vaex.convert
        path_output = convert if isinstance(convert, str) else vaex.convert._convert_name(filename_or_buffer)
        vaex.convert.convert_csv(
            path_input=filename_or_buffer, fs_options_input=fs_options, fs_input=fs,
            path_output=path_output, fs_options_output=fs_options, fs_output=fs,
            chunk_size=chunk_size,
            copy_index=copy_index,
            progress=progress,
            **kwargs
        )
        return open(path_output, fs_options=fs_options, fs=fs)


def _read_csv_read(filename_or_buffer, copy_index, chunk_size, fs_options={}, fs=None, **kwargs):
    import pandas as pd
    if not chunk_size:
        with vaex.file.open(filename_or_buffer, fs_options=fs_options, fs=fs, for_arrow=True) as f:
            if "compression" not in kwargs:
                try:
                    path = vaex.file.stringyfy(filename_or_buffer)
                except:
                    path = None
                if path:
                    parts = path.rsplit('.', 3)
                    if len(parts) == 3:
                        # we need to do infer here, because pandas does not look at the fileobj.name
                        # to infer the compression
                        extension_to_compression = {"gz": "gzip", "bz2": "bz2", "zip": "zip", "xz": "xz"}
                        if parts[-1] in extension_to_compression:
                            kwargs = {"compression": extension_to_compression[parts[-1]], **kwargs}
            full_df = pd.read_csv(f, **kwargs)
            return from_pandas(full_df, copy_index=copy_index)
    else:
        def iterator():
            chunk_iterator = pd.read_csv(filename_or_buffer, chunksize=chunk_size, **kwargs)
            for chunk_df in chunk_iterator:
                yield from_pandas(chunk_df, copy_index=copy_index)
        return iterator()


def read_csv(filepath_or_buffer, **kwargs):
    '''Alias to from_csv.'''
    return from_csv(filepath_or_buffer, **kwargs)

aliases = vaex.settings.aliases


def connect(url, **kwargs):
    """Connect to hostname supporting the vaex web api.

    :param str hostname: hostname or ip address of server
    :rtype: vaex.server.client.Client
    """
    # dispatch to vaex.server package
    from vaex.server import connect
    return connect(url, **kwargs)

def example():
    '''Result of an N-body simulation of the accretion of 33 satellite galaxies into a Milky Way dark matter halo.

    Data was greated by Helmi & de Zeeuw 2000.
    The data contains the position (x, y, z), velocitie (vx, vy, vz), the energy (E),
    the angular momentum (L, Lz) and iron content (FeH) of the particles.

    :rtype: DataFrame
    '''
    return vaex.datasets.helmi_simulation_data()



# there are kept for backwards compatibility
# TODO: remove in vaex v5?

def set_log_level_debug(loggers=["vaex"]):
    """set log level to debug"""
    vaex.logging.set_log_level_debug(loggers)


def set_log_level_info(loggers=["vaex"]):
    """set log level to info"""
    vaex.logging.set_log_level_info(loggers)


def set_log_level_warning(loggers=["vaex"]):
    """set log level to warning"""
    vaex.logging.set_log_level_warning(loggers)


def set_log_level_exception(loggers=["vaex"]):
    """set log level to exception/error"""
    vaex.logging.set_log_level_error(loggers)


def set_log_level_off():
    """Disabled logging"""
    vaex.logging.set_log_level_off()


import_script = os.path.expanduser("~/.vaex/vaex_import.py")
if os.path.exists(import_script):
    try:
        with open(import_script) as f:
            code = compile(f.read(), import_script, 'exec')
            exec(code)
    except:
        import traceback
        traceback.print_stack()


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


for entry in entry_points(group='vaex.namespace'):
    logger.warning('(DEPRECATED, use vaex.dataframe.accessor) adding vaex namespace: ' + entry.name)
    try:
        add_namespace = entry.load()
        add_namespace()
    except Exception:
        logger.exception('issue loading ' + entry.name)

_lazy_accessors_map = {}


class _lazy_accessor(object):
    def __init__(self, name, scope, loader, lazy_accessors):
        """When adding an accessor geo.cone, scope=='geo', name='cone', scope may be falsy"""
        self.loader = loader
        self.name = name
        self.scope = scope
        self.lazy_accessors = lazy_accessors

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
            if fullname in self.lazy_accessors:
                for name, scope, loader, lazy_accessors in self.lazy_accessors[fullname]:
                    assert fullname == scope
                    setattr(cls, name, property(_lazy_accessor(name, scope, loader, lazy_accessors)))
        return obj.__dict__[self.name]


def _add_lazy_accessor(name, loader, target_class=vaex.dataframe.DataFrame):
    """Internal use see tests/internal/accessor_test.py for usage

    This enables us to have df.foo.bar accessors that lazily loads the modules.
    """
    parts = name.split('.')
    if target_class not in _lazy_accessors_map:
        _lazy_accessors_map[target_class] = {}
    lazy_accessors = _lazy_accessors_map[target_class]
    if len(parts) == 1:
        setattr(target_class, parts[0], property(_lazy_accessor(name, None, loader, lazy_accessors)))
    else:
        scope = ".".join(parts[:-1])
        if scope not in lazy_accessors:
            lazy_accessors[scope] = []
        lazy_accessors[scope].append((parts[-1], scope, loader, lazy_accessors))


for entry in entry_points(group='vaex.dataframe.accessor'):
    logger.debug('adding vaex accessor: ' + entry.name)
    def loader(entry=entry):
        return entry.load()
    _add_lazy_accessor(entry.name, loader)


for entry in entry_points(group='vaex.expression.accessor'):
    logger.debug('adding vaex expression accessor: ' + entry.name)
    def loader(entry=entry):
        return entry.load()
    _add_lazy_accessor(entry.name, loader, vaex.expression.Expression)


for entry in entry_points(group='vaex.plugin'):
    try:
        module_name = entry.module 
    except AttributeError:
        module_name = entry.module_name
    if module_name == 'vaex_arrow.opener':
        # if vaex_arrow package is installed, we ignore it
        continue
    logger.debug('adding vaex plugin: ' + entry.name)
    try:
        add_namespace = entry.load()
        add_namespace()
    except Exception:
        logger.exception('issue loading ' + entry.name)


def concat(dfs, resolver='flexible') -> vaex.dataframe.DataFrame:
    '''Concatenate a list of DataFrames.

    :param resolver: How to resolve schema conflicts, see :meth:`DataFrame.concat`.
    '''
    df, *tail = dfs
    return df.concat(*tail, resolver=resolver)

def vrange(start, stop, step=1, dtype='f8'):
    """Creates a virtual column which is the equivalent of numpy.arange, but uses 0 memory

    :param int start: Start of interval. The interval includes this value.
    :param int stop: End of interval. The interval does not include this value,
    :param int step: Spacing between values.
    :dtype: The preferred dtype for the column.
    """
    from .column import ColumnVirtualRange
    return ColumnVirtualRange(start, stop, step, dtype)

def vconstant(value, length, dtype=None, chunk_size=1024):
    """Creates a virtual column with constant values, which uses 0 memory.

    :param value: The value with which to fill the column
    :param length: The length of the column, i.e. the number of rows it should contain.
    :param dtype: The preferred dtype for the column.
    :param chunk_size: Could be used to optimize the performance (evaluation) of this column.
    """
    from .column import ColumnVirtualConstant
    return ColumnVirtualConstant(value=value, length=length, dtype=dtype, chunk_size=chunk_size)

def string_column(strings):
    import pyarrow as pa
    return pa.array(strings)


def dtype(type):
    '''Creates a Vaex DataType based on a NumPy or Arrow type'''
    return vaex.datatype.DataType(type)

def dtype_of(ar) -> vaex.datatype.DataType:
    '''Creates a Vaex DataType from a NumPy or Arrow array'''
    if isinstance(ar, vaex.dataset.Column):
        return dtype(ar.dtype)
    elif vaex.array_types.is_arrow_array(ar):
        return dtype(ar.type)
    elif vaex.array_types.is_numpy_array(ar) or isinstance(ar, vaex.column.supported_column_types):
        return dtype(ar.dtype)
    else:
        raise TypeError(f'{ar} is not a an Arrow or NumPy array')


class RowLimitException(ValueError):
    pass

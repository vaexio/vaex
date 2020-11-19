import re
import logging
import os

import vaex
import vaex.file
import vaex.cache

log = logging.getLogger('vaex.cache')


def _convert_name(filenames, shuffle=False, suffix=None):
    '''Convert a filename (or list of) to a filename with .hdf5 and optionally a -shuffle or other suffix'''
    if not isinstance(filenames, (list, tuple)):
        filenames = [filenames]
    filenames = [vaex.file.stringyfy(f) for f in filenames]
    base = filenames[0]
    if shuffle:
        base += '-shuffle'
    if suffix:
        base += suffix
    if len(filenames) > 1:
        return base + "_and_{}_more.hdf5".format(len(filenames)-1)
    else:
        return base + ".hdf5"


def convert(path_input, fs_options_input, path_output, fs_options_output, *args, **kwargs):
    @vaex.cache.output_file(
        path_input=path_input, fs_options_input=fs_options_input,
        path_output=path_output, fs_options_output=fs_options_output)
    def cached_output(*args, **kwargs):
        naked_path, _ = vaex.file.split_options(path_input)
        _, ext, _ = vaex.file.split_ext(path_input)
        if ext == '.csv' or naked_path.endswith(".csv.bz2"):  # special support for csv.. should probably approach it a different way
            if fs_options_output:
                raise ValueError(f'No fs_options support for output/convert')
            if 'chunk_size' not in kwargs:
                # make it memory efficient by default
                kwargs['chunk_size'] = 5_000_000
            _from_csv_convert_and_read(path_input, maybe_convert_path=path_output, fs_options=fs_options_input, **kwargs)
        else:
            ds = vaex.dataset.open(path_input, fs_options=fs_options_input, *args, **kwargs)
            if ds is not None:
                df = vaex.from_dataset(ds)
                df.export(path_output)
    cached_output(*args, **kwargs)



def _from_csv_convert_and_read(filename_or_buffer, maybe_convert_path, chunk_size, fs_options, copy_index=False, **kwargs):
    # figure out the CSV file path
    if isinstance(maybe_convert_path, str):
        csv_path = re.sub(r'\.hdf5$', '', str(maybe_convert_path), flags=re.IGNORECASE)
    elif isinstance(filename_or_buffer, str):
        csv_path = filename_or_buffer
    else:
        raise ValueError('Cannot derive filename to use for converted HDF5 file, '
                         'please specify it using convert="my.csv.hdf5"')

    combined_hdf5 = _convert_name(csv_path)

    # convert CSV chunks to separate HDF5 files
    import pandas as pd
    converted_paths = []
    csv_reader = pd.read_csv(filename_or_buffer, chunksize=chunk_size, **kwargs)
    for i, df_pandas in enumerate(csv_reader):
        df = vaex.from_pandas(df_pandas, copy_index=copy_index)
        filename_hdf5 = _convert_name(csv_path, suffix='_chunk%d' % i)
        df.export_hdf5(filename_hdf5)
        converted_paths.append(filename_hdf5)
        log.info('saved chunk #%d to %s' % (i, filename_hdf5))

    # combine chunks into one HDF5 file
    if len(converted_paths) == 1:
        # no need to merge several HDF5 files
        os.rename(converted_paths[0], combined_hdf5)
    else:
        log.info('converting %d chunks into single HDF5 file %s' % (len(converted_paths), combined_hdf5))
        dfs = [vaex.open(p) for p in converted_paths]
        df_combined = vaex.concat(dfs)
        df_combined.export_hdf5(combined_hdf5)

        log.info('deleting %d chunk files' % len(converted_paths))
        for df, df_path in zip(dfs, converted_paths):
            try:
                df.close()
                os.remove(df_path)
            except Exception as e:
                log.error('Could not close or delete intermediate hdf5 file %s used to convert %s to hdf5: %s' % (
                    df_path, csv_path, e))

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


def convert(path_input, fs_options_input, fs_input, path_output, fs_options_output, fs_output, *args, **kwargs):
    @vaex.cache.output_file(
        path_input=path_input, fs_options_input=fs_options_input, fs_input=fs_input,
        path_output=path_output, fs_options_output=fs_options_output, fs_output=fs_output)
    def cached_output(*args, **kwargs):
        ds = vaex.dataset.open(path_input, fs_options=fs_options_input, fs=fs_input, *args, **kwargs)
        if ds is not None:
            df = vaex.from_dataset(ds)
            df.export(path_output, fs_options=fs_options_output, fs=fs_output)
    cached_output(*args, **kwargs)


def convert_csv(path_input, fs_options_input, fs_input, path_output, fs_options_output, fs_output, *args, **kwargs):
    @vaex.cache.output_file(
        path_input=path_input, fs_options_input=fs_options_input, fs_input=fs_input,
        path_output=path_output, fs_options_output=fs_options_output, fs_output=fs_output)
    def cached_output(*args, **kwargs):
        if fs_options_output:
            raise ValueError(f'No fs_options support for output/convert')
        if 'chunk_size' not in kwargs:
            # make it memory efficient by default
            kwargs['chunk_size'] = 5_000_000
        _from_csv_convert_and_read(path_input, maybe_convert_path=path_output, fs_options=fs_options_input, fs=fs_input, **kwargs)
    cached_output(*args, **kwargs)


def _from_csv_convert_and_read(filename_or_buffer, maybe_convert_path, chunk_size, fs_options, fs=None, copy_index=False, **kwargs):
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
    with vaex.file.open(filename_or_buffer, fs_options=fs_options, fs=fs, for_arrow=True) as f:
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


def main(argv):
    import argparse
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument('--verbose', '-v', action='count', default=0)
    parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")
    parser.add_argument('--list', '-l', default=False, action='store_true', help="list columns of input")
    parser.add_argument('--progress', help="show progress (default: %(default)s)", default=True, action='store_true')
    parser.add_argument('--no-progress', dest="progress", action='store_false')
    parser.add_argument('--no-delete', help="Delete file on failure (default: %(default)s)", dest='delete', default=True, action='store_false')
    parser.add_argument('--shuffle', "-s", dest="shuffle", action='store_true', default=False)
    parser.add_argument('--sort', dest="sort", default=None)
    parser.add_argument('--fraction', "-f", dest="fraction", type=float, default=1.0, help="fraction of input dataset to export")
    parser.add_argument('--filter', dest="filter", default=None, help="filter to apply before exporting")
    parser.add_argument("input", help="input source or file, when prefixed with @ it is assumed to be a text file with a file list (one file per line)")
    parser.add_argument("output", help="output file (ends in .hdf5)")
    parser.add_argument("columns", help="list of columns to export (or all when empty)", nargs="*")

    args = parser.parse_args(argv[1:])

    verbosity = ["ERROR", "WARNING", "INFO", "DEBUG"]
    logging.getLogger("vaex").setLevel(verbosity[min(3, args.verbose)])
    if args.input[0] == "@":
        inputs = open(args.input[1:]).readlines()
        df = vaex.open_many(inputs)
    else:
        df = vaex.open(args.input)

    if df:
        df.set_active_fraction(args.fraction)
    if args.list:
        print("\n".join(df.get_column_names()))
    else:
        if args.columns:
            all_columns = df.get_column_names()
            columns = args.columns
            for column in columns:
                if column not in all_columns:
                    # if not args.quiet:
                    print("column %r does not exist, run with --list or -l to list all columns" % column)
                    return 1
            df = df[columns]
        else:
            columns = df.get_column_names()

        if not args.quiet:
            print("exporting %d rows and %d columns" % (len(df), len(columns)))
            print("columns: " + " ".join(columns))

        if args.filter:
            df = df.filter(args.filter)
        if args.sort:
            df = df.sort(args.sort)
        try:
            df.export(args.output, progress=args.progress)
            if not args.quiet:
                print("\noutput to %s" % os.path.abspath(args.output))
            df.close()
        except:
            if not args.quiet:
                print("\nfailed to write to%s" % os.path.abspath(args.output))
            if args.delete:
                if args.delete:
                    os.remove(args.output)
                    print("\ndeleted output %s (pass --no-delete to avoid that)" % os.path.abspath(args.output))
            raise


    return 0

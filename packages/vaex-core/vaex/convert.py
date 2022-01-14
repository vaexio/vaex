import logging
import os
import sys

import vaex
import vaex.cache
import vaex.file
import vaex.progress

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


def convert(path_input, fs_options_input, fs_input, path_output, fs_options_output, fs_output, progress=None, *args, **kwargs):
    @vaex.cache.output_file(
        path_input=path_input, fs_options_input=fs_options_input, fs_input=fs_input,
        path_output=path_output, fs_options_output=fs_options_output, fs_output=fs_output)
    def cached_output(*args, **kwargs):
        ds = vaex.dataset.open(path_input, fs_options=fs_options_input, fs=fs_input, *args, **kwargs)
        if ds is not None:
            df = vaex.from_dataset(ds)
            df.export(path_output, fs_options=fs_options_output, fs=fs_output, progress=progress)
    cached_output(*args, **kwargs)


def convert_csv(path_input, fs_options_input, fs_input, path_output, fs_options_output, fs_output, progress=None, *args, **kwargs):
    @vaex.cache.output_file(
        path_input=path_input, fs_options_input=fs_options_input, fs_input=fs_input,
        path_output=path_output, fs_options_output=fs_options_output, fs_output=fs_output)
    def cached_output(*args, **kwargs):
        if fs_options_output:
            raise ValueError(f'No fs_options support for output/convert')
        if 'chunk_size' not in kwargs:
            # make it memory efficient by default
            kwargs['chunk_size'] = 5_000_000
        _from_csv_convert_and_read(path_input, path_output=path_output, fs_options=fs_options_input, fs=fs_input, progress=progress, **kwargs)
    cached_output(*args, **kwargs)


def _from_csv_convert_and_read(filename_or_buffer, path_output, chunk_size, fs_options, fs=None, copy_index=False, progress=None, **kwargs):
    # figure out the CSV file path
    csv_path = vaex.file.stringyfy(filename_or_buffer)
    path_output_bare, ext, _ = vaex.file.split_ext(path_output)

    combined_hdf5 = _convert_name(csv_path)

    # convert CSV chunks to separate HDF5 files
    import pandas as pd
    converted_paths = []
    # we don't have indeterminate progress bars, so we cast it to truethy
    progress = bool(progress) if progress is not None else False
    if progress:
        print("Converting csv to chunk files")
    with vaex.file.open(filename_or_buffer, fs_options=fs_options, fs=fs, for_arrow=True) as f:
        csv_reader = pd.read_csv(filename_or_buffer, chunksize=chunk_size, **kwargs)
        for i, df_pandas in enumerate(csv_reader):
            df = vaex.from_pandas(df_pandas, copy_index=copy_index)
            chunk_name = f'{path_output_bare}_chunk_{i}{ext}'
            df.export(chunk_name)
            converted_paths.append(chunk_name)
            log.info('saved chunk #%d to %s' % (i, chunk_name))
            if progress:
                print("Saved chunk #%d to %s" % (i, chunk_name))

    # combine chunks into one HDF5 file
    if len(converted_paths) == 1:
        # no need to merge several HDF5 files
        os.rename(converted_paths[0], path_output)
    else:
        if progress:
            print('Converting %d chunks into single file %s' % (len(converted_paths), path_output))
        log.info('converting %d chunks into single file %s' % (len(converted_paths), path_output))
        dfs = [vaex.open(p) for p in converted_paths]
        df_combined = vaex.concat(dfs)
        df_combined.export(path_output, progress=progress)

        log.info('deleting %d chunk files' % len(converted_paths))
        for df, df_path in zip(dfs, converted_paths):
            try:
                df.close()
                os.remove(df_path)
            except Exception as e:
                log.error('Could not close or delete intermediate file %s used to convert %s to single file: %s', (df_path, csv_path, path_output))


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
    parser.add_argument('--optimize', help="run df.optimize.categorize, downcast and cast float64 to float32 before exporting (default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--categorize', help="run df.optimize.categorize before exporting (default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--downcast', help="run df.optimize.downcast before exporting (default: %(default)s)", default=False, action='store_true')
    parser.add_argument('--downcast-float', help="Downcast float64 to float32 (default: %(default)s)", default=False, action='store_true')
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

        with vaex.progress.tree(title="export preprocess"):
            if args.filter:
                df = df.filter(args.filter)
            if args.sort:
                df = df.sort(args.sort)
            if args.shuffle:
                df = df.shuffle()
            if args.optimize or args.categorize:
                df = df.optimize.categorize()
            if args.optimize or args.downcast:
                df = df.optimize.downcast(float64=args.optimize or args.downcast_float)
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

if __name__ == "__main__":
    main(sys.argv)

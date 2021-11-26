# -*- coding: utf-8 -*-
import sys
import os
import platform

darwin = "darwin" in platform.system().lower()
frozen = getattr(sys, 'frozen', False)
if frozen and len(sys.argv) > 1 and sys.argv[1].startswith("-psn"):  # is the 'app' is opened in osx, just stars without arguments
    dirname = os.path.join(os.path.dirname(sys.argv[0]), "..", "..", "..")
    os.chdir(dirname)


if darwin and frozen:  # on newer osx versions we get a lot of broken pipes when writing to stdout
    directory = os.path.expanduser("~/.vaex")
    if not os.path.exists(directory):
        os.makedirs(directory)
    sys.stdout = open(os.path.expanduser('~/.vaex/stdout.txt'), 'w')
    sys.stderr = open(os.path.expanduser('~/.vaex/stderr.txt'), 'w')


# print darwin, platform.system()


usage = """usage veax [-h] {webserver,convert,...}

optional arguments:
  -h, --help          show this help message and exit

positional arguments:
    webserver           start the vaex webserver
    convert             convert datasets
    benchmark           run benchmarks
    meta                import/export meta data
    alias               manage aliases
    stat                print statistics/info about dataset
    test                run unittests
    open                tests opening of a file (will return exit error on failure)
    settings            control and view settings
    ...                 anything else will start up the gui, see usage below

Examples:
$ vaex convert s3://vaex/taxi/nyc_taxi_2015_mini.parquet taxi.hdf5
$ vaex convert taxi.hdf5 taxi.csv

"""


def main(args=None):
    if args is None:
        args = sys.argv
    if frozen and len(args) > 1 and args[1].startswith("-psn"):  # is the 'app' is opened in osx, just start without arguments
        import vaex.ui.main
        vaex.ui.main.main([])
    else:
        if len(args) > 1 and args[1] in ["-h", "--help"]:
            print(usage)
            sys.exit(0)
        if len(args) > 1 and args[1] == "version":
            import vaex.version
            if frozen:
                extra = " (build on %s using Python %s)" % (vaex.version.osname, sys.version)
            else:
                extra = " (using Python %s)" % (sys.version)
            print(vaex.__full_name__ + extra)
        elif len(args) > 1 and args[1] in ["webserver", "server"]:
            import vaex.server.server
            vaex.server.server.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "benchmark":
            import vaex.benchmark
            vaex.benchmark.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "convert":
            import vaex.convert
            vaex.convert.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "meta":
            import vaex.meta
            vaex.meta.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "alias":
            import vaex.misc_cmdline
            vaex.misc_cmdline.alias_main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "stat":
            import vaex.misc_cmdline
            vaex.misc_cmdline.stat_main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "open":
            sys.exit(open_main([os.path.basename(args[0]) + " " + args[1]] + args[2:]))
        elif len(args) > 1 and args[1] == "test":
            import vaex.test.__main__
            vaex.test.__main__.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "settings":
            import vaex.settings
            vaex.settings._main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        else:
            print(usage)
            sys.exit(0)

def open_main(argv):
    import argparse
    parser = argparse.ArgumentParser(argv[0])
    parser.add_argument('--verbose', '-v', default=False, action='store_true', help="give extra output")
    parser.add_argument('--quiet', '-q', default=False, action='store_true', help="do not output anything")
    parser.add_argument('--dry-run', '-n', default=False, action='store_true', help="do not actually execute commands (like delete)")
    parser.add_argument('--delete', help="Delete file when reading fails", default=False, action='store_true')
    parser.add_argument("input", help="list of files to try to open", nargs="*")

    args = parser.parse_args(argv[1:])
    import vaex
    import vaex.file
    failed = False
    if args.verbose:
        print(f"Checking files {', '.join(args.input)}")
    for path in args.input:
        try:
            vaex.open(path)
        except BaseException as e:
            failed = True
            if not args.quiet:
                print(e)
            if args.delete:
                if not args.quiet:
                    print(f'rm {path}')
                if not args.dry_run:
                    try:
                        vaex.file.remove(path)
                    except FileNotFoundError:
                        pass
    if args.verbose:
        if failed:
            print("Oops, had issues opening some files")
        else:
            print("All files could be opened")
    return 123 if failed else 0

if __name__ == "__main__":
    main()

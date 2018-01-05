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
    ...                 anything else will start up the gui, see usage below


All other cases will start up the gui:
usage: vaex [input expression [expression ...] [key=value ...]
            [-|--|+|++] input expression...]

input        input file or url (if an url is provided, a dataset name should follow)
expression   expressions to plot, for instance "x**2", "y" "x+y"
-|--|+|++    - will open a new window, no input is required
             -- will open a new window with a new dataset (input is required)
             + will add a new layer, no input is required
             ++ will add a new layer with a new dataset (input is required)
key=value    will set properties of the plot, see a list of values below

there can be 1, 2 or 3 expressions, resulting in a 1d histogram, a 2d density plot or a
 3d volume rendering window.

key=value options:
amplitude          (string) expression for how the histogram translates to an amplitude,
                    available grids:
                       counts: N dimensional grid containing the counts
                       weighted: N dimensional grid containing the sum of
                            the weighted values
weight             (string) expression for the weight field
grid_size          (integer, power of 2), defines the size of the histogram grid, example
                       grid_size=64 grid_size=129
vector_grid_size   similar, for the vector histogram
vx, vy, vz         (string) expressions for the x,y and z component of the vector grid


Examples:
# single window, showing log(counts+1) as amplitude
$ vaex example.hdf5 x y amplitude="log(counts+1)'

# adding a second window, showing x vs z
vaex example.hdf5 x y amplitude="log(counts+1)" - x z amplitude="log(counts+1)"

# adding a second window with a new dataset
vaex example.hdf5 x y amplitude="log(counts+1)" -- example2.hdf5 x z amplitude="log(counts+1)"

# single window, with an extra layer
vaex example.hdf5 x y amplitude="log(counts+1)" + y x amplitude="log(counts+1)"

# single window, with an extra layer from a new dataset
vaex example.hdf5 x y amplitude="log(counts+1)" ++ example2.hdf5 x y amplitude="log(counts+1)"

see more examples at http://TODO

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
        elif len(args) > 1 and args[1] == "webserver":
            import vaex.webserver
            vaex.webserver.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "benchmark":
            import vaex.benchmark
            vaex.benchmark.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "convert":
            import vaex.export
            vaex.export.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "meta":
            import vaex.meta
            vaex.meta.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "alias":
            import vaex.misc_cmdline
            vaex.misc_cmdline.alias_main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "stat":
            import vaex.misc_cmdline
            vaex.misc_cmdline.stat_main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        elif len(args) > 1 and args[1] == "test":
            import vaex.test.__main__
            vaex.test.__main__.main([os.path.basename(args[0]) + " " + args[1]] + args[2:])
        else:
            import vaex.ui.main
            vaex.ui.main.main(args[1:])


if __name__ == "__main__":
    main()

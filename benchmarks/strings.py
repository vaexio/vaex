"""
Benchark ran on my laptop:
python benchmarks/strings.py -n8 --npandas=6.5

To benchmark on AWS, I did the following
 * spin up a large h1.8xlarge machine

from the shell:

sudo apt-get update
sudo apt-get install g++
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda install -c conda-forge numpy scipy notebook matplotlib -y
# if you want master
# git clone --recursive https://github.com/vaexio/vaex/
# cd vaex
# pip install .
pip install vaex-core vaex-hdf5 vaex-arrow 

mkdir ramdisk
sudo mount -t tmpfs -o size=50g tmpfs ramdisk

python strings.py -n9 --npandas=6.5 --filename=ramdisk/test.hdf5 --partitions=100
"""
import timeit
import vaex
import numpy as np
import dask.dataframe as dd
import dask
import json
import sys
import pandas as pd
import os
import time
import argparse
import multiprocessing

filter_name = sys.argv[1] if len(sys.argv) > 1 else None

default_filename = 'string_benchmark.hdf5'

argv = sys.argv
parser = argparse.ArgumentParser(argv[0])
parser.add_argument('--number', "-n", dest="n", type=float, default=7, help="log number of rows to use")
parser.add_argument('--nmax', type=int, default=9, help="number of rows for test dataset")
parser.add_argument('--partitions', type=int, default=multiprocessing.cpu_count()*2, help="number of partitions to split (default: 2x number cores)")
parser.add_argument('--npandas', dest="npandas", type=float, default=7, help="number of rows to use for pandas")
parser.add_argument('--filter', dest="filter", default=None, help="filter for benchmark")
parser.add_argument('--filename', default=default_filename, help='filename to use for benchmark export/reading')
args = parser.parse_args(argv[1:])

use_dask = False


if not os.path.exists(args.filename):
    x = np.arange(0, int(10**args.nmax))
    xs = x.astype(str)
    s = xs#vaex.string_column(xs)
    df_vaex = vaex.from_arrays(x=s, s=s)
    df_vaex.export(args.filename, progress=True, shuffle=True)

df = vaex.open(args.filename)
df_vaex = df[0:int(10**args.n)]
df_vaex.executor.buffer_size = len(df_vaex)//args.partitions
df_pandas = df[:int(10**args.npandas)].to_pandas_df()

if use_dask:
    df_dask = dd.from_pandas(df_pandas, npartitions=4)
timings = {}
def mytimeit(expr, N, scope):
    times = []
    for i in range(N):
        t0 = time.time()
        eval(expr, scope)
        times.append(time.time() - t0)
    return times
def test(name, expr):
    if args.filter and args.filter not in name:
        return
    vaex_expr = expr + ".nop()"
    N = 5
    # t = timeit.Timer(vaex_expr, globals={'df': df_vaex})
    # print(name, expr)
    # t_vaex = min(t.repeat(3, N))/N/(10**args.n)
    # print("\tvaex", t_vaex)

    t_vaex = min(mytimeit(vaex_expr, N, scope={'df': df_vaex}))/(10**args.n)
    print(name, expr, '2')
    print("\tvaex", t_vaex)

    # t = timeit.Timer(expr, globals={'df': df_pandas})
    # t_pandas = min(t.repeat(3, N))/N/(10**args.npandas)
    t_pandas = min(mytimeit(expr, N, scope={'df': df_pandas}))/(10**args.npandas)
    print("\tpandas", t_pandas, t_pandas/t_vaex)

    timings[name] = {'vaex': t_vaex, 'pandas': t_pandas}

    if use_dask:
        t_dask = min(mytimeit(expr+".compute()", N, scope={'df': df_dask}))/(10**args.npandas)
        print("\tdask", t_dask)
        timings[name]['dask'] = t_dask

        with dask.config.set(scheduler='processes'):
            t_dask = min(mytimeit(expr+".compute()", N, scope={'df': df_dask}))/(10**args.npandas)
            print("\tdask_process", t_dask)
            timings[name]['dask_process'] = t_dask

if __name__ == '__main__':

    test('capitalize', 'df.s.str.capitalize()')
    test('cat', 'df.s.str.cat(df.s)')
    test('contains', 'df.s.str.contains("9", regex=False)')
    test('contains(regex)', 'df.s.str.contains("9", regex=True)')
    test('count', 'df.s.str.count("9")')
    test('endswith', 'df.s.str.endswith("9")')
    test('find', 'df.s.str.find("4")')
    test('get', 'df.s.str.get(1)')
    # test('index', 'df.s.str.index(1, 3)') TODO?
    test('split+join', 'df.s.str.split(".").str.join("-")')
    test('len', 'df.s.str.len()')
    test('ljust', 'df.s.str.ljust(10)')
    test('lower', 'df.s.str.lower()')
    test('lstrip', 'df.s.str.lstrip("9")')
    test('match', 'df.s.str.match("1.*")')
    test('pad', 'df.s.str.pad(10)')
    test('repeat', 'df.s.str.repeat(2)')
    test('replace(default)', 'df.s.str.replace("123", "321")')
    test('replace(no regex)', 'df.s.str.replace("123", "321", regex=False)')
    test('replace(regex)', 'df.s.str.replace("1?[45]4", "1004", regex=True)')
    test('rfind', 'df.s.str.rfind("4")')
    test('rjust', 'df.s.str.rjust(10)')
    test('rstrip', 'df.s.str.rstrip("9")')
    test('slice', 'df.s.str.slice(1, 3)')
    test('split', 'df.s.str.split(".")')
    test('startswith', 'df.s.str.startswith("9")')
    test('strip', 'df.s.str.strip("0")') # issues?
    test('title', 'df.s.str.title()')
    test('upper', 'df.s.str.upper()')
    test('zfill', 'df.s.str.zfill(10)')

    for name, values in timings.items():
        tv = values['vaex']
        tp = values['pandas']
        print(name)
        if use_dask:
            td = values['dask']
            print('\t', tv, tp, tp/tv, td/tv, tp/td)
        else:
            print('\t', tv, tp, tp/tv)
   
    previous_timings = {}
    fn = "timings.json"
    if os.path.exists(fn):
        with open(fn, "r") as f:
            previous_timings = json.load(f)

    previous_timings.update(timings)
    timings = previous_timings
    print(timings)
    fn = "timings.json"
    with open(fn, "w") as f:
        json.dump(timings, f)
    print('write', fn)
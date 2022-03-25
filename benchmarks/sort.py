import vaex
import os
from benchmarks.fixtures import generate_strings
from benchmarks.fixtures import generate_numerical


def time_sort_strings(N):
    df = vaex.open(generate_strings())
    df = df[:N]
    df.sort(df.s)

time_sort_strings.params = [10**k for k in [5, 6]]

def time_sort_ints(N, T):
    df = vaex.open(generate_numerical())
    df = df[:N]
    df.sort(f'i8_{T}')

time_sort_ints.params = [10**k for k in [5, 6]], ['100', '1M']

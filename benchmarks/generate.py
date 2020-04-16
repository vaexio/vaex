import os

import numpy as np

import vaex.utils


def generate_numerical():
    # x and y floating points (float64), gaussian distriubyted
    # x4 and y4 for float32 performance testing
    # iB_10 contains 10 random distinct integers between 5 and 15, where B=[1,2,4,8], for int8 to int64
    # iB_1K contains 1000 random distinct integers between 5 and 1005, where B=[2,4,8] for int16 to int64
    # iB_1M contains 1_000_000 random distinct integers between 5 and 1_000_005, where B=[4,8] for int32 to int64
    filename = 'numerical.hdf5'
    # goes to ~/.veax/benchmarks
    path = os.path.join(vaex.utils.get_private_dir('benchmarks'), filename)
    print(path)
    nmax = 9
    if not os.path.exists(path):
        N = int(10 ** nmax)
        x, y = np.random.normal(0, 1, (2, N))
        i8 = vaex.vrange(0, N, dtype=np.int64)
        df = vaex.from_arrays(i8=i8, x=x, y=y)
        df['x4'] = df['x'].astype('float32')
        df['y4'] = df['y'].astype('float32')

        for size in [1, 2, 4]:
            typename = f'i{size}'
            df[typename] = df['i8'].astype(typename)

        i10 = np.random.randint(5, 10 + 5, N, dtype=np.int64)
        i1K = np.random.randint(5, 1000 + 5, N, dtype=np.int64)
        i1M = np.random.randint(5, 1_000_000 + 5, N, dtype=np.int64)

        df['i8_10'] = i10
        df['i8_1K'] = i1K
        df['i8_1M'] = i1M
        for byte_size in [1, 2, 4]:
            typename = f'i{byte_size}'
            df[f'i{byte_size}_10'] = df['i8_10'].astype(typename)
            if byte_size >= 2:
                df[f'i{byte_size}_1K'] = df['i8_1K'].astype(typename)
            if byte_size >= 4:
                df[f'i{byte_size}_1M'] = df['i8_1M'].astype(typename)
        df.export(path, progress=True, virtual=True)

if __name__ == '__main__':
    generate_numerical()


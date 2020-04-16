import vaex
import numpy as np
import os
import generate


class Aggregates:
    params = ([10**7, 10**8, 5*10**8, 10**9],)
    param_names = ['N']

    def setup_cache(self):
        generate.generate_numerical()

    def setup(self, N):
        basedir = vaex.utils.get_private_dir('benchmarks')
        self.df = vaex.open(os.path.join(basedir, 'numerical.hdf5'))[:N]
        self.df.categorize(self.df.i8_10, list(range(5, 15)), check=False)
        self.df.categorize(self.df.i4_10, list(range(5, 15)), check=False)
        self.df.categorize(self.df.i2_10, list(range(5, 15)), check=False)
        self.df.categorize(self.df.i1_10, list(range(5, 15)), check=False)


class Stats(Aggregates):
    def time_count_star(self, N):
        self.df.count()

    def time_mean_x(self, N):
        self.df.x.mean()

    def time_mean_x4(self, N):
        self.df.x4.mean()


class BinByCat10(Aggregates):
    params = Aggregates.params + ([1, 2, 4, 8],)
    param_names = ['N', 'B']

    def setup(self, N, B):
        super().setup(N)

    def time_binby_iB_10(self, N, B):
        self.df.count(binby=f'i{B}_10')


class BinByCat1K(Aggregates):
    params = Aggregates.params + ([2, 4, 8],)
    param_names = ['N', 'B']

    def setup(self, N, B):
        super().setup(N)

    def time_binby_iB_1K(self, N, B):
        self.df.count(binby=f'i{B}_1K')


class BinByCat1M(Aggregates):
    params = Aggregates.params + ([4, 8],)
    param_names = ['N', 'B']

    def setup(self, N, B):
        super().setup(N)

    def time_binby_iB_1K(self, N, B):
        self.df.count(binby=f'i{B}_1M')


class BinBy1d(Aggregates):
    def time_count_star(self, N):
        self.df.count()

    def time_count_star_binby128(self, N):
        self.df.count(binby='x', limits=[-1, 1], shape=128)

    def time_count_x_binby128(self, N):
        self.df.count('x', binby='x', limits=[-1, 1], shape=128)


class BinBy2d(Aggregates):
    def time_count_star(self, N):
        self.df.count(binby=[self.df.x, self.df.y], limits=[-1, -1], shape=128)

    def time_count_star_x4(self, N):
        self.df.count(binby=[self.df.x4, self.df.y4], limits=[-1, -1], shape=128)

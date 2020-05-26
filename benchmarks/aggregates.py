import vaex

from benchmarks.fixtures import generate_numerical


class Aggregates:
    pretty_name = "Performance of aggregates: stats, binby etc"
    version = "1"

    params = ([10**7, 5*10**7, 10**8],)
    param_names = ['N']

    def setup_cache(self):
        # ensure the dataframe is generated
        generate_numerical()

    def setup(self, N):
        self.df = vaex.open(generate_numerical())[:N]
        self.df.categorize(self.df.i8_10, min_value=5, max_value=15, inplace=True)
        self.df.categorize(self.df.i4_10, min_value=5, max_value=15, inplace=True)
        self.df.categorize(self.df.i2_10, min_value=5, max_value=15, inplace=True)
        self.df.categorize(self.df.i1_10, min_value=5, max_value=15, inplace=True)

        self.df.categorize(self.df.i8_1K, min_value=5, max_value=1_000+5, inplace=True)
        self.df.categorize(self.df.i4_1K, min_value=5, max_value=1_000+5, inplace=True)
        self.df.categorize(self.df.i2_1K, min_value=5, max_value=1_000+5, inplace=True)
        # self.df.categorize(self.df.i1_1K, min_value=5, max_value=1_000+5)

        self.df.categorize(self.df.i8_1M, min_value=5, max_value=1_000_000+5, inplace=True)
        self.df.categorize(self.df.i4_1M, min_value=5, max_value=1_000_000+5, inplace=True)
        # self.df.categorize(self.df.i2_1M, min_value=5, max_value=1_000_000+5)
        # self.df.categorize(self.df.i1_1M, min_value=5, max_value=1_000_000+5)


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

    def time_binby_iB_1M(self, N, B):
        self.df.count(binby=f'i{B}_1M')


class GroupByCat10(Aggregates):
    params = Aggregates.params + ([1, 2, 4, 8],)
    param_names = ['N', 'B']

    def setup(self, N, B):
        super().setup(N)

    def time_binby_iB_10(self, N, B):
        self.df.groupby(f'i{B}_10', agg='count')


class GroupByCat1K(Aggregates):
    params = Aggregates.params + ([2, 4, 8],)
    param_names = ['N', 'B']

    def setup(self, N, B):
        super().setup(N)

    def time_binby_iB_1M(self, N, B):
        self.df.groupby(f'i{B}_1k', agg='count')


class GroupByCat1M(Aggregates):
    params = Aggregates.params + ([4, 8],)
    param_names = ['N', 'B']

    def setup(self, N, B):
        super().setup(N)

    def time_binby_iB_1K(self, N, B):
        self.df.groupby(f'i{B}_1M', agg='count')


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

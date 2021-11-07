import vaex

from benchmarks.fixtures import generate_numerical


class Filter:
    pretty_name = "Filter benchmarks"
    version = "1"

    params = ([10**7, 10**8],)
    param_names = ['N']

    def setup_cache(self):
        # ensure the dataframe is generated
        generate_numerical()

    def setup(self, N):
        self.df = vaex.open(generate_numerical())[:N]
        self.df['i4_1M_POT'] = self.df['i4_1M'] * 2 ** 8
        self.dff = self.df[(self.df.x > 0) & (self.df.y < 0)]
        len(self.dff)  # fill cache

    def time_filter_and_head(self, N):
        self.dff.head()

    def time_filter_and_tail(self, N):
        self.dff.tail()

    def time_filter_count(self, N):
        self.dff.count('i1_100')  # but this should be super fast, since the cache


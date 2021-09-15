import vaex

from benchmarks.fixtures import generate_numerical


class GroupbyBasic:
    pretty_name = "Groupby benchmarks"
    version = "1"

    params = ([10**7, 5*10**7, 10**8],)
    param_names = ['N']

    def setup_cache(self):
        # ensure the dataframe is generated
        generate_numerical()

    def setup(self, N):
        self.df = vaex.open(generate_numerical())[:N]
        self.df['i4_1M_POT'] = self.df['i4_1M'] * 2 ** 8

    def time_count_i1_100(self, N):
        df = self.df.groupby(['i1_100'], agg='count')

    def time_count_i4_1M(self, N):
        df = self.df.groupby(['i4_1M'], agg='count')

    def time_count_i4_1M_POT(self, N):
        df = self.df.groupby(['i4_1M_POT'], agg='count')


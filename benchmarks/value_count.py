import vaex

from benchmarks.fixtures import generate_numerical, generate_strings2


class ValueCounts:
    pretty_name = "Value counts benchmarks"
    version = "1"

    params = ([10**7, 10**8, 10**9],)
    # params = ([10**9],)
    param_names = ['N']

    def setup_cache(self):
        # ensure the dataframe is generated
        generate_numerical()

    def setup(self, N):
        self.df = vaex.open(generate_numerical())[:N]
        self.df['i4_1M_POT'] = self.df['i4_1M'] * 2 ** 8
        self.df_s = vaex.open(generate_strings2())[:N]

    def time_value_counts_string_100(self, N):
        self.df_s['s100'].value_counts()

    def time_value_counts_string_1M(self, N):
        self.df_s['s1M'].value_counts()

    def time_value_counts_i1_100(self, N):
        self.df['i1_100'].value_counts()

    def time_value_counts_i8_100(self, N):
        self.df['i8_100'].value_counts()

    def time_value_counts_i4_1M(self, N):
        self.df['i4_1M'].value_counts()

    def time_value_counts_i4_1M_POT(self, N):
        self.df['i4_1M_POT'].value_counts()

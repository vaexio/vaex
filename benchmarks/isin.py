from benchmarks.fixtures import generate_numerical, generate_strings
import vaex


class IsIn:
    pretty_name = "Performance of isin"
    version = "1"

    params = ([10**7, 5*10**7, 10**8], (1, 10, 100, 1_000, 1_000_000))
    param_names = ['N', 'M']

    def setup_cache(self):
        generate_numerical()
        generate_strings()

    def setup(self, N, M):
        self.df_num = vaex.open(generate_numerical())[:N]
        self.df_str = vaex.open(generate_strings())[:N]

    def time_isin_i8_1M(self, N, M):
        df = self.df_num
        values = df.sample(M)['i8_1M'].values
        df['i8_1M'].isin(values).sum()

    def time_isin_str(self, N, M):
        df = self.df_str
        values = df.sample(M)['s'].values
        df['s'].isin(values).sum()

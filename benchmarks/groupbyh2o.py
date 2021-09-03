import vaex

from benchmarks.fixtures import generate_numerical

check = False

# in h2o we have (with n=1e8 and k=1e2)
# id1 - str with 1e2 different values
# id2 - str with 1e2 different values
# id3 - str with 1e2 different values
# id4 - int with 1e2 different values
# id5 - int with 1e2 different values
# id6 - int with 1e6 different values

class GroupBySetup:
    pretty_name = "Groupby benchmarks - H2O inspired"
    version = "1"

    params = ([10**7, 5*10**7, 10**8],)
    param_names = ['N']

    def setup_cache(self):
        # ensure the dataframe is generated
        generate_numerical()

    def setup(self, N):
        df = self.df = vaex.open(generate_numerical())[:N]
        df['id1'] = df['i1_100']
        df['id2'] = df['i1_100']
        df['id3'] = df['i4_1M']
        df['id4'] = df['i1_100']
        df['id5'] = df['i1_100']
        df['id6'] = df['i4_1M']
        df['v1'] = df['i1_10']
        df['v2'] = df['i1_10']
        df['v3'] = df['x4']


class GroupbyH2O(GroupBySetup):
    def time_question_01(self, N):
        df = self.df.groupby(['id1']).agg({'v1': 'sum'})
        if check:
            chk_sum_cols = ['v1']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_02(self, N):
        df = self.df.groupby(['id1', 'id2']).agg({'v1': 'sum'})
        if check:
            chk_sum_cols=['v1']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_03(self, N):
        df = self.df.groupby(['id3']).agg({'v1': 'sum', 'v3': 'mean'})
        if check:
            chk_sum_cols=['v1', 'v3']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_04(self, N):
        df = self.df.groupby(['id4']).agg({'v1':'mean', 'v2':'mean', 'v3':'mean'})
        if check:
            chk_sum_cols=['v1', 'v2', 'v3']
            [df[col].sum() for col in chk_sum_cols]

    def time_question_05(self, N):
        df = self.df.groupby(['id6']).agg({'v1': 'sum', 'v2': 'sum', 'v3': 'sum'})
        if check:
            chk_sum_cols=['v1', 'v2', 'v3']
            [df[col].sum() for col in chk_sum_cols]

    # def time_question_6(self, N):
    #     self.df.groupby(['i4_10','i4_1K']).agg({'x': ['median','std']})

    def time_question_07(self, N):
        df = self.df.groupby(['id3']).agg({'v1': 'max', 'v2': 'min'})
        df['range_x_y'] = df['v1'] - df['v2']
        if check:
            chk_sum_cols=['range_v1_v2']
            [df[col].sum() for col in chk_sum_cols]

    # def time_question_8(self, N):
    #     self.df[['id6', 'v3']].sort('v3', ascending=False).groupby(['id6'])
    #     if check:
    #         chk_sum_cols=['v1']
    #         [df[col].sum() for col in chk_sum_cols]

    # def time_question_9(self.N):
    #     Not implemented yet

    def time_question_10(self, N):
        df = self.df.groupby(['id1', 'id2', 'id3', 'id4', 'id5', 'id6']).agg({'v3':'sum', 'v1':'count'})
        if check:
            chk_sum_cols=['v3', 'v1']
            [df[col].sum() for col in chk_sum_cols]

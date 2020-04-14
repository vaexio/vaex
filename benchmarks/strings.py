import os

import numpy as np
import vaex


class StringsSuite:
    pretty_name = "Performance of string operations in DataFrames"
    version = "1"
    timeout = 3600  # in seconds

    param_names = [
        'n',  # log number of rows to use
        'partitions',  # number of partitions to split
    ]
    params = ([6], [24])

    dataframe_nmax = 7  # log number of rows for test dataset
    dataframe_filename = 'string_benchmark.hdf5'

    def setup_cache(self):
        if not os.path.exists(self.dataframe_filename):
            x = np.arange(0, int(10 ** self.dataframe_nmax))
            s = x.astype(str)
            df_vaex = vaex.from_arrays(x=s, s=s)
            df_vaex.export(self.dataframe_filename, progress=True, shuffle=True)

    def setup(self, n, partitions):
        df = vaex.open(self.dataframe_filename)
        df_vaex = df[0:int(10 ** n)]
        df_vaex.executor.buffer_size = len(df_vaex) // partitions
        self.df = df_vaex

    def time_capitalize(self, n, partitions):
        self.df.s.str.capitalize().nop()

    def time_cat(self, n, partitions):
        self.df.s.str.cat(self.df.s).nop()

    def time_contains(self, n, partitions):
        self.df.s.str.contains("9", regex=False).nop()

    def time_contains_regex(self, n, partitions):
        self.df.s.str.contains("9", regex=True).nop()

    def time_count(self, n, partitions):
        self.df.s.str.count("9").nop()

    def time_endswith(self, n, partitions):
        self.df.s.str.endswith("9").nop()

    def time_find(self, n, partitions):
        self.df.s.str.find("4").nop()

    def time_get(self, n, partitions):
        self.df.s.str.get(1).nop()

    def time_split_and_join(self, n, partitions):
        self.df.s.str.split(".").str.join("-").nop()

    def time_len(self, n, partitions):
        self.df.s.str.len().nop()

    def time_ljust(self, n, partitions):
        self.df.s.str.ljust(10).nop()

    def time_lower(self, n, partitions):
        self.df.s.str.lower().nop()

    def time_lstrip(self, n, partitions):
        self.df.s.str.lstrip("9").nop()

    def time_match(self, n, partitions):
        self.df.s.str.match("1.*").nop()

    def time_pad(self, n, partitions):
        self.df.s.str.pad(10).nop()

    def time_repeat(self, n, partitions):
        self.df.s.str.repeat(2).nop()

    def time_replace_default(self, n, partitions):
        self.df.s.str.replace("123", "321").nop()

    def time_replace_no_regex(self, n, partitions):
        self.df.s.str.replace("123", "321", regex=False).nop()

    def time_replace_regex(self, n, partitions):
        self.df.s.str.replace("1?[45]4", "1004", regex=True).nop()

    def time_rfind(self, n, partitions):
        self.df.s.str.rfind("4").nop()

    def time_rjust(self, n, partitions):
        self.df.s.str.rjust(10).nop()

    def time_rstrip(self, n, partitions):
        self.df.s.str.rstrip("9").nop()

    def time_slice(self, n, partitions):
        self.df.s.str.slice(1, 3).nop()

    def time_split(self, n, partitions):
        self.df.s.str.split(".").nop()

    def time_startswith(self, n, partitions):
        self.df.s.str.startswith("9").nop()

    def time_strip(self, n, partitions):
        self.df.s.str.strip("0").nop()

    def time_title(self, n, partitions):
        self.df.s.str.title().nop()

    def time_upper(self, n, partitions):
        self.df.s.str.upper().nop()

    def time_zfill(self, n, partitions):
        self.df.s.str.zfill(10).nop()

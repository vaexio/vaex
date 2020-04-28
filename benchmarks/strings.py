import vaex

from benchmarks.fixtures import generate_strings


class Strings:
    pretty_name = "Performance of string operations"
    version = "1"
    timeout = 3600  # in seconds

    param_names = [
        'N',  # number of rows to use
    ]
    params = ([10**7])

    def setup_cache(self):
        # ensure the dataframe is generated
        generate_strings()

    def setup(self, N):
        partitions = 24
        df = vaex.open(generate_strings())
        df_vaex = df[0:int(N)]
        df_vaex.executor.buffer_size = len(df_vaex) // partitions
        self.df = df_vaex

    def time_capitalize(self, N):
        self.df.s.str.capitalize().nop()

    def time_cat(self, N):
        self.df.s.str.cat(self.df.s).nop()

    def time_contains(self, N):
        self.df.s.str.contains("9", regex=False).nop()

    def time_contains_regex(self, N):
        self.df.s.str.contains("9", regex=True).nop()

    def time_count(self, N):
        self.df.s.str.count("9").nop()

    def time_endswith(self, N):
        self.df.s.str.endswith("9").nop()

    def time_find(self, N):
        self.df.s.str.find("4").nop()

    def time_get(self, N):
        self.df.s.str.get(1).nop()

    def time_split_and_join(self, N):
        self.df.s.str.split(".").str.join("-").nop()

    def time_len(self, N):
        self.df.s.str.len().nop()

    def time_ljust(self, N):
        self.df.s.str.ljust(10).nop()

    def time_lower(self, N):
        self.df.s.str.lower().nop()

    def time_lstrip(self, N):
        self.df.s.str.lstrip("9").nop()

    def time_match(self, N):
        self.df.s.str.match("1.*").nop()

    def time_pad(self, N):
        self.df.s.str.pad(10).nop()

    def time_repeat(self, N):
        self.df.s.str.repeat(2).nop()

    def time_replace_default(self, N):
        self.df.s.str.replace("123", "321").nop()

    def time_replace_no_regex(self, N):
        self.df.s.str.replace("123", "321", regex=False).nop()

    def time_replace_regex(self, N):
        self.df.s.str.replace("1?[45]4", "1004", regex=True).nop()

    def time_rfind(self, N):
        self.df.s.str.rfind("4").nop()

    def time_rjust(self, N):
        self.df.s.str.rjust(10).nop()

    def time_rstrip(self, N):
        self.df.s.str.rstrip("9").nop()

    def time_slice(self, N):
        self.df.s.str.slice(1, 3).nop()

    def time_split(self, N):
        self.df.s.str.split(".").nop()

    def time_startswith(self, N):
        self.df.s.str.startswith("9").nop()

    def time_strip(self, N):
        self.df.s.str.strip("0").nop()

    def time_title(self, N):
        self.df.s.str.title().nop()

    def time_upper(self, N):
        self.df.s.str.upper().nop()

    def time_zfill(self, N):
        self.df.s.str.zfill(10).nop()

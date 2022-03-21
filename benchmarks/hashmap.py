import vaex
from benchmarks.fixtures import generate_strings
from benchmarks.fixtures import generate_numerical



class hashmap:
    pretty_name = "hashmap benchmarks"
    version = "1"

    def setup(self):
        dfs = vaex.open(generate_strings())
        Nmax_strings = 1_000_000
        self.dfs_small = dfs[:Nmax_strings]
        self.hms = self.dfs_small._hash_map_unique("s")

    def time_strings_create(self):
        self.dfs_small._hash_map_unique('s')

    def time_strings_keys(self):
        self.hms.keys()


if __name__ == "__main__":
    bench = hashmap()
    bench.setup()
    bench.time_strings_create()
    # time_hashmap_strings_create()
import os
from datetime import datetime

import pandas as pd
import vaex
from vaex.datasets import nyctaxi_yellow_2015_jan

# download and read Pandas CSV
nyctaxi_yellow_2015_jan.download()
csv_size = os.path.getsize(nyctaxi_yellow_2015_jan.filenames[0])
df = pd.read_csv(nyctaxi_yellow_2015_jan.filenames[0])

# convert to Vaex
start = datetime.now()
vdf = vaex.from_pandas(df)
duration = datetime.now() - start

print('it took {} to convert {:,} rows ({:.1f} Gb), which is {:,} rows per second'.format(
    duration, len(df), csv_size / 1024. / 1024 / 1024, int(len(df) / duration.total_seconds())))

# Last result when running on:
# 2.8 GHz Quad-Core Intel Core i7; 16 GB 1600 MHz DDR3:
# it took 0:00:04.454153 to convert 12,748,986 rows (1.8 Gb), which is 2,862,269 rows per second

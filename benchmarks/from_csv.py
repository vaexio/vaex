import glob
import os
from datetime import datetime

import vaex


def test_open_two_big_csv_convert():
    big_and_biggest_csv = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1*.csv'
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1_01.csv.hdf5')
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1.csv.hdf5')
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1.csv_and_1_more.hdf5')

    start = datetime.now()
    df = vaex.open(big_and_biggest_csv, convert=True)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(
        duration, df.length(), int(df.length() / duration.total_seconds())))

    # with default chunk_size and pandas:
    #   it took 0:07:28.072395 to convert 52,126,928 rows, which is 116,335 rows per second


def test_open_several_medium_csv_convert():
    csv_glob = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_*.csv'
    for path in glob.glob(csv_glob):
        os.remove(path + '.hdf5')
    os.remove('/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_0.csv_and_3_more.hdf5')

    start = datetime.now()
    df = vaex.open(csv_glob, convert=True)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(
        duration, df.length(), int(df.length() / duration.total_seconds())))
    assert df.length() == 3_999_999

    # with default chunk_size and pandas:
    #   it took 0:00:20.083716 to convert 3,999,999 rows, which is 199,166 rows per second


def test_from_big_csv_read():
    # csv = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_0.csv'
    csv = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    # csv = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1.csv'

    start = datetime.now()
    read_length = 0
    # for df in vaex.from_csv(csv, chunk_size=2_000_000):
    #     read_length += len(df)
    read_length += len(vaex.from_csv(csv))
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(
        duration, read_length, int(read_length / duration.total_seconds())))
    assert read_length == 7_667_792
    # assert df.length() == 44_459_136

    # with different chunk sizes or without chunks at all, more or less the same time:
    #   it took 0:00:17.112055 to convert 7,667,792 rows, which is 448,092 rows per second


def test_from_big_csv_convert():
    # csv = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_0.csv'
    csv = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    # csv = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-h1.csv'
    os.remove(csv + '.hdf5')

    start = datetime.now()
    df = vaex.from_csv(csv, convert=True)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(
        duration, df.length(), int(df.length() / duration.total_seconds())))
    assert df.length() == 7_667_792
    # assert df.length() == 44459136

    # with default chunk_size and pandas:
    #   it took 0:00:41.300364 to convert 7,667,792 rows, which is 185,659 rows per second


def test_read_csv_and_convert():
    test_path = '/Users/byaminov/fun/datasets/test_yellow_tripdata/yellow_tripdata_2019-01_*.csv'

    # remove pre-converted files
    import os
    import glob
    for hdf_file in glob.glob(test_path.replace('.csv', '.hdf5')):
        print('deleting %s' % hdf_file)
        os.remove(hdf_file)

    start = datetime.now()
    df = vaex.read_csv_and_convert(test_path, copy_index=False)
    duration = datetime.now() - start
    print('it took {} to convert {:,} rows, which is {:,} rows per second'.format(
        duration, df.length(), df.length() / duration.total_seconds()))
    assert df.length() == 3999999


def test_pandas_read_csv_chunked():
    test_path = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    import pandas as pd

    start = datetime.now()
    n_read = 0
    for df in pd.read_csv(test_path, chunksize=1_000_000):
        n_read += len(df)
    duration = datetime.now() - start
    print('it took {} to read {:,} rows, which is {:,} rows per second'.format(
        duration, n_read, int(n_read / duration.total_seconds())))
    assert n_read == 7_667_792

    # it took 0:00:12.945616 to read 7,667,792 rows, which is 592,308 rows per second


def test_arrow_read_csv_chunked():
    test_path = '/Users/byaminov/fun/datasets/yellow_tripdata_2019-01.csv'
    from pyarrow import csv

    start = datetime.now()
    table = csv.read_csv(test_path)
    duration = datetime.now() - start
    print('it took {} to read {:,} rows, which is {:,} rows per second'.format(
        duration, len(table), int(len(table) / duration.total_seconds())))
    assert len(table) == 7_667_792

    # it took 0:00:02.553384 to read 7,667,792 rows, which is 3,002,992 rows per second

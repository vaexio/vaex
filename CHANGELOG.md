# vaex 4.9.1

# vaex-core 4.9.1
   * Fix
      * When aggregation leads to arrow data, non-dense binners failed (e.g. vaex.agg.list) [#2017](https://github.com/vaexio/vaex/pull/2017)
      * Filtering by boolean column would miss the column as dependency [#2016](https://github.com/vaexio/vaex/pull/2016)

# vaex 4.9.0

# vaex-core 4.9.0
   * Features
      * Progress bar for percentile_approx and median_approx [#1889](https://github.com/vaexio/vaex/pull/1889)
      * Better casting of strings to datetime [#1920](https://github.com/vaexio/vaex/pull/1920)
      * We better support numpy scalars now, and more arrow time units. [#1921](https://github.com/vaexio/vaex/pull/1921)
      * Allow sorting by strings, multiple columns and multiple directions [#1963](https://github.com/vaexio/vaex/pull/1963)
      * Support JSON in df.export [#1974](https://github.com/vaexio/vaex/pull/1974)
      * New/better aggregators
        * first/last use different type 'sort column. [#1848](https://github.com/vaexio/vaex/pull/1848)
        * Skew and kurtosis [#1946](https://github.com/vaexio/vaex/pull/1946)
        * List aggregator [#1987](https://github.com/vaexio/vaex/pull/1987)
      * Pre-sort by the grouping columns in df.groupby (better performance) [#1990](https://github.com/vaexio/vaex/pull/1990)
   * Performance
      * No copy of hashmap and GIL release [#1893](https://github.com/vaexio/vaex/pull/1893) [#1961](https://github.com/vaexio/vaex/pull/1961)
      * Store strings in hashmap in arrow array, making map.key_array() faster [#1976](https://github.com/vaexio/vaex/pull/1976)
   * Fix
      * Respect row_limit when the groupby is dense [#1894](https://github.com/vaexio/vaex/pull/1894)
      * Fingerprint collision possible if filter uses virtual column [#1949](https://github.com/vaexio/vaex/pull/1949)
      * Apply with filtered data could give wrong dtypes [#1936](https://github.com/vaexio/vaex/pull/1936)
      * Strings array growing failed when first string was zero length [#1956](https://github.com/vaexio/vaex/pull/1956)
      * Use less processes for when using multiprocessing. [#1979](https://github.com/vaexio/vaex/pull/1979)
      * Support chunked arrays and empty chunks in value counts. [#1958](https://github.com/vaexio/vaex/pull/1958) [#1975](https://github.com/vaexio/vaex/pull/1975)
      * Allow renaming of function, to make join use with functions without name collisions. [#1966](https://github.com/vaexio/vaex/pull/1966)
      * Join would fail if the rhs had no columns besides the join one [#2010](https://github.com/vaexio/vaex/pull/2010)
      * hdf5 export fails for concat df with missing columns [#1493](https://github.com/vaexio/vaex/pull/1493)
      * Allow `col` as column name [#1992](https://github.com/vaexio/vaex/issues/1992)



# vaex 4.8.0
## vaex-core 4.8.0
   * Features
      * Multiple example datasets provided in `vaex.datasets` [#1317](https://github.com/vaexio/vaex/pull/1317)
      * We do not use asyncio for the default sync execute path [#1783](https://github.com/vaexio/vaex/pull/1783)
      * Executor works with asyncio with multiple tasks [#1784]https://github.com/vaexio/vaex/pull/1784)
      * Auto execute context manager makes vaex behave normal with await [#1785](https://github.com/vaexio/vaex/pull/1785)
      * Support exporting arrow and parquet to file like objects [#1790](https://github.com/vaexio/vaex/pull/1790)
      * Put lock files in $VAEX_HOME/lock [#1797](https://github.com/vaexio/vaex/pull/1797)
      * Show progress when converting the included datasets [#1798](https://github.com/vaexio/vaex/pull/1798)
      * Limit and limit_raise for unique and nunique  [#1801](https://github.com/vaexio/vaex/pull/1801)
      * Lazy ordinal encode [#1813](https://github.com/vaexio/vaex/pull/1813)
      * Configure logging using settings system[#1811](https://github.com/vaexio/vaex/pull/1811)
      * Export to JSON [#1789](https://github.com/vaexio/vaex/pull/1789)
      * Progress bar can be configured using settings system [#1815](https://github.com/vaexio/vaex/pull/1815)
      * fillna and fillmissing should upcast integers when needed [#1869](https://github.com/vaexio/vaex/pull/1869)
   * Performance
      * Moved mutex use to the C++ layer to avoid GIL issues [#1847](https://github.com/vaexio/vaex/pull/1847)
      * Many refactors to improve performance [#1863](https://github.com/vaexio/vaex/pull/1863) [#1869](https://github.com/vaexio/vaex/pull/1869)
   * Fix
      * Support empty parquet and arrow files [#1791](https://github.com/vaexio/vaex/pull/1791)
      * Keep virtual column order when renaming/dropping to not break state transfer [#1788](https://github.com/vaexio/vaex/pull/1788)
      * blake3 compatibility issues [#1818](https://github.com/vaexio/vaex/pull/1818) [db527a6](https://github.com/vaexio/vaex/commit/db527a6942db6ee74d97f1f1e8e5ddb3e8978f0c)
      * Avoid frozendict 2.2.0 which can segfault on Python 3.6[#1856](https://github.com/vaexio/vaex/pull/1856)
      * Use label instead of expression for non-ident column names in binby [#1842](https://github.com/vaexio/vaex/pull/1842)
   * Development
      * Use cmake/scikit-build [1847](https://github.com/vaexio/vaex/pull/1847) [92af1b1](https://github.com/vaexio/vaex/commit/92af1b1fab55dcc36c93e327495ac239c3fef772) [ad88d4b](https://github.com/vaexio/vaex/commit/ad88d4b2525c9fda7798c685985d9391a6b498a5)

## vaex-hdf5 0.12.0
   * Features
      * Support storing Arrow Dictionary encoded/categoricals in hdf5 [#1814](https://github.com/vaexio/vaex/pull/1814)

## vaex-ml 0.17.0
   Requires vaex-core 4.8.0 for the `vaex.datasets.iris()`

## vaex-server 0.8.1
   Made compatible with Python 3.6


# vaex 4.7.0
## vaex-core 4.7.0
   * Features
      * Allow casting integers to timedelta64 type [#1741](https://github.com/vaexio/vaex/pull/1741)
      * When a single task can fail, other can continue [#1762](https://github.com/vaexio/vaex/pull/1762)
      * Improved rich progress bar support [#1771](https://github.com/vaexio/vaex/pull/1771)
      * vaex.from_records to build a dataframe from a list of dicts [#1767](https://github.com/vaexio/vaex/pull/1767)
      * Settings in Vaex can be configured in a uniform way [#1743](https://github.com/vaexio/vaex/pull/1743)
      * Unique for datetime64 and timedelta64 expressions [#1016](https://github.com/vaexio/vaex/pull/1016)
      * Copy argument for binby, similar to groupby [4e7fd8e](https://github.com/vaexio/vaex/commit/4e7fd8e154c151323410cc1bedec96dd8a9667cb)
   * Performance
      * Improve performance for filtered dataframes [1685](https://github.com/vaexio/vaex/pull/1685)
   * Fixes
      * S3: endpoint override combined with globbing [#1739](https://github.com/vaexio/vaex/pull/1739)
      * Support having filtered and unfiltered tasks in 1 pass over the data [#1761](https://github.com/vaexio/vaex/pull/1761)
      * Continue next tasks even when old ones are cancelled [#1769](https://github.com/vaexio/vaex/pull/1769)
      * Handle empty arrow files [#1773](https://github.com/vaexio/vaex/pull/1773)
      * Evaluate and evaluate_iterator did not work for empty filtered dataframes [#1775](https://github.com/vaexio/vaex/pull/1775)

## vaex-hdf5 0.11.1
   * Features
      * do not track times to have deterministic output (useful for lineage/hash output) [#1772](https://github.com/vaexio/vaex/pull/1772)

## vaex-ml 0.16
Requires vaex-core 4.7 for uniform settings

## vaex-server 0.8
Requires vaex-core 4.7 for uniform settings

## vaex-jupyter 0.7
Requires vaex-core 4.7 for uniform settings
   * Features
      * Editor widget for settings [#1743](https://github.com/vaexio/vaex/pull/1743)

## vaex-viz 0.5.1
  * Fixes
   * Histogram method on expression to propagate kwargs [#1757](https://github.com/vaexio/vaex/pull/1757)


# vaex 4.6.0
## vaex-core 4.6.0
   * Features
      * OSX Metal support for jitting expressions [#584](https://github.com/vaexio/vaex/pull/584)
      * Improved progress support, including Rich progress bars [#1738](https://github.com/vaexio/vaex/pull/1738)
      * Control number of columns and rows being printed [#1672](https://github.com/vaexio/vaex/pull/1672)
      * Groupby with regular bins (similar to binby) [#1589](https://github.com/vaexio/vaex/pull/1589)
      * Groupby with a limited number of values, and 'OTHERS' [#1641](https://github.com/vaexio/vaex/pull/1641)
      * New aggregators: vaex.agg.any and vaex.agg.all [#1630](https://github.com/vaexio/vaex/pull/1630)
      * Better API for correlation and mutual information [#536](https://github.com/vaexio/vaex/pull/536)
      * Materialize datasets columns for better performance of non-memory mapping files (e.g. parquet) [#1625](https://github.com/vaexio/vaex/pull/1625)
      * Avoid using nest_asyncio [#1546](https://github.com/vaexio/vaex/pull/1546)
      * Multi level cache support (e.g. memory and disk) [#1580](https://github.com/vaexio/vaex/pull/1580)
      * Do not mutate dataframe when comparing dates. [#1584](https://github.com/vaexio/vaex/pull/1584)
   * Performance
      * Fingerprint for tasks are more stable when the dataframe changes, but not the task description, for more cache hits. [#1627](https://github.com/vaexio/vaex/pull/1627)
      * Faster conversion between Arrow and NumPy [#1625](https://github.com/vaexio/vaex/pull/1625)
      * Cache sparse-finding/combining of high-d groupby [#1588](https://github.com/vaexio/vaex/pull/1588)
      * Allow (lazy) math and computations with aggregators [#1612](https://github.com/vaexio/vaex/pull/1612)
      * Less passes over the data when multiple dataframes use the same dataset [#1594](https://github.com/vaexio/vaex/pull/1594)
      * Share evaluation of expressions of selections [#1594](https://github.com/vaexio/vaex/pull/1594)
      * Delay support for groupby [#1594](https://github.com/vaexio/vaex/pull/1594)
   * Fixes
      * Missing values in groupby were not well supported [#1637](https://github.com/vaexio/vaex/pull/1637)
      * Groupby over boolean [#1632](https://github.com/vaexio/vaex/pull/1632)
      * Negative periods for diff and shift [#1608](https://github.com/vaexio/vaex/pull/1608)
      * Arrow timestamp promotion during concatenation [#1551](https://github.com/vaexio/vaex/pull/1551)

## vaex-server 0.7
Requires vaex-core 4.6

## vaex-ml 0.15
Requires vaex-core 4.6
   * Performance
      * Dot product with many columns does not use expressions, but dedicated function [#1671](https://github.com/vaexio/vaex/pull/1671)

# vaex 4.5.0
## vaex-core 4.5.1
   * Features
      * Filelocks for multi process convert=True cooperation [#1573](https://github.com/vaexio/vaex/pull/1573)
   * Performance
      * Fingerprint speedups [#1574](https://github.com/vaexio/vaex/pull/1574)
      * Expression.nunique asked unique for Python list (slow) [#1576](https://github.com/vaexio/vaex/pull/1576)
      * Groupby was slow for particular data (with low bits 0) [#1571](https://github.com/vaexio/vaex/pull/1571)
      * Blob encoding is faster due to blake3 [#1575](https://github.com/vaexio/vaex/pull/1575)
      * Stop iterating over dataset when exception occurs when computing [#1577](https://github.com/vaexio/vaex/pull/1577)

## vaex-core 4.5.0
   * Features
      * Protect file creation parts with lock files [#1541](https://github.com/vaexio/vaex/pull/1541)
      * Expression.str.extract to extract parts of strings using regex to a struct [#1423](https://github.com/vaexio/vaex/pull/1423)
   * Performance
      * We now cache Expression.nunique() [#1565](https://github.com/vaexio/vaex/pull/1565)
      * Hashmaps memory is shared among threads (does not scale linear with number of threads), and avoids a merge phase [#1525](https://github.com/vaexio/vaex/pull/1525)
      * Hashmaps serialize efficiently [#1525](https://github.com/vaexio/vaex/pull/1525)
      * Avoid falling back to eval to get the dtype [#1514](https://github.com/vaexio/vaex/pull/1514)

## vaex-hdf5 0.10.0
   * Features
      * Write higher dimensional arrays to hdf5 files [#1563](https://github.com/vaexio/vaex/pull/1563)

## vaex-ml 0.14.0
   * Features
      * MultiHotEncoder [#1537](https://github.com/vaexio/vaex/pull/1537)
      * Various ML metrics [#1529](https://github.com/vaexio/vaex/pull/1529)

## vaex-astro 0.9
   Requires vaex 4.5.0 due to private API change.

## vaex-server 0.6.1
   * Fixes
      * Missing imports (now checked in CI) [#1516](https://github.com/vaexio/vaex/pull/1516)

## vaex-contrib 0.1.0
   * Features
      * Import from and export to Google BigQuery [#1470](https://github.com/vaexio/vaex/pull/1470)


# vaex 4.4.0
## vaex-core 4.4.0
   * Performance
      * Reuse filter data when slicing a dataframe [#1287](https://github.com/vaexio/vaex/pull/1287)
      * Faster astype('str') [#1411](https://github.com/vaexio/vaex/pull/1411)
      * Task refactor, which allows for more caching [#1433](https://github.com/vaexio/vaex/pull/1433)
   * Features
      * df.to_records() for output to JSON [#1364](https://github.com/vaexio/vaex/pull/1364)
      * df.dt.quarter and df.dt.halfyear [#1441](https://github.com/vaexio/vaex/pull/1364)https://github.com/vaexio/vaex/pull/1441)
      * Arrow struct support [#1447](https://github.com/vaexio/vaex/pull/1364)https://github.com/vaexio/vaex/pull/1447)
   * Fixes
      * df.concat did not copy functions  [#1287](https://github.com/vaexio/vaex/pull/1287)
      * Dropping columns when name was invalid identifier [#1434](https://github.com/vaexio/vaex/pull/1434)
      * Old dates wrapped due to negative ints and casting to unsigned [#1504](https://github.com/vaexio/vaex/pull/1504)
      * Timestamp to NumPy datetime64 would ignore units [#1513](https://github.com/vaexio/vaex/pull/1513)
      * Thread unsafety could trigger issues in Plotly dash [#1512](https://github.com/vaexio/vaex/pull/1512)

## vaex-server 0.6.0
   * Complete refactor, now using FastAPI by default [#1300](https://github.com/vaexio/vaex/pull/1300)

## vaex-ml 0.13.0
   * Tensorflow/keras support [#1510](https://github.com/vaexio/vaex/pull/1510)

## vaex-hdf5 0.9.0
   * Features
      * Support writing/reading from custom hdf5 groups [#1439](https://github.com/vaexio/vaex/pull/1510)
      * Support laying out an empty hdf5 file for writing [#1439](https://github.com/vaexio/vaex/pull/1510)
   * Fixes
      * File order close issue on Windows [#1479](https://github.com/vaexio/vaex/pull/1479)

# vaex 4.3.0
## vaex-core 4.3.0
   * Performance
      * Reuse filter data when slicing a dataframe [#1287](https://github.com/vaexio/vaex/pull/1287)
   * Features
      * Cache task results, with support for Redis and diskcache [#1393](https://github.com/vaexio/vaex/pull/1393)
      * df.func.stack for stacking columns into Nd arrays [#1287](https://github.com/vaexio/vaex/pull/1287)
      * Sliding windows / shift / diff / sum [#1287](https://github.com/vaexio/vaex/pull/1287)
      * Embed join/groupby/shift in dataset (opt in via df._future(), will be default in vaex v5) [#1287](https://github.com/vaexio/vaex/pull/1287)
      * df.fingerprint() - a cross runtime unique key for caching [#1287](https://github.com/vaexio/vaex/pull/1287)
      * limit rows in groupby using early stop [#1391](https://github.com/vaexio/vaex/pull/1391)
      * Compare date columns to string values formatted in ISO 8601 format 621a341b54f9b4112f24e2ffd86612753df19fef
   * Fixes
      * df.concat did not copy functions  [#1287](https://github.com/vaexio/vaex/pull/1287)
      * Filters with column name equals to function names a159777e2dc13ec762914c51c8b5550efec5f845

# vaex 4.2.0
## vaex-core 4.2.0
   * Performance
      * Perform groupby in a sparse way for less memory usage/performance (up to 250x faster) [#1381](https://github.com/vaexio/vaex/pull/1381)
   * Features
      * Sorted groupby [#1339](https://github.com/vaexio/vaex/pull/1339)
   * Fixes
      * Proper use of logging framework [#1384](https://github.com/vaexio/vaex/pull/1384)
      * Aggregating with 'count' would ignore custom names [#1345](https://github.com/vaexio/vaex/pull/1345)
      * Join supports datetime column
## vaex-ml 0.12.0
   * Features
      * River integration [#1256](https://github.com/vaexio/vaex/pull/1256)
      * Random projections [#1305](https://github.com/vaexio/vaex/pull/1256)
      * Incremental PCA [#1289](https://github.com/vaexio/vaex/pull/1289)
## vaex-server 0.4.1
   * Features
      * SSL support 5dc29edd5b15eb4e1fe9c6981c67edd477481484

# vaex 4.1.0 (2021-Mar-18)
## vaex-core 4.1.0
   * Features
      *  groupby datetime support [#1265](https://github.com/vaexio/vaex/pull/1265)
   * Fixes
      * Improved fsspec support [#1268](https://github.com/vaexio/vaex/pull/1268)
   * Performance
      * df.extract() uses mask instead of indices 398b682fe9042b3336120e9013e15bbd638620ed


# vaex 4.0.0 (2021-Mar-9)
   * Breaking changes:
      * Arrow is now a core dependency, vaex-arrow is deprecated. All methods that return string, will return Arrow arrays [#517](https://github.com/vaexio/vaex/pull/517)
      * Opening an .arrow file will expose the arrays as Apache Arrow arrays, not numpy arrays. [#984](https://github.com/vaexio/vaex/pull/984)
      * Columns (e.g. df.column['x']) may now return a ColumnProxy, instead of the original data, slice it [:] to get the underlying data (or call .to_numpy()/to_arrow() or try converting it with np.array(..) or pa.array(..)). [#993](https://github.com/vaexio/vaex/pull/993)
      * All plot methods went into the df.viz accessor [#923](https://github.com/vaexio/vaex/pull/923)

## vaex-arrow (DEPRECATED)
   This is now part of vaex-core.

## vaex-astro 0.8.0
  * Requirement changed to vaex-core >=4,<5

## vaex-core 4.0.0
   * Fixes
      * Repeated dropna/dropnan/dropmissing could report cached length. [#874](https://github.com/vaexio/vaex/pull/874)
      * Trimming concatenated columns. [#860](https://github.com/vaexio/vaex/pull/860)
      * percentile_approx works for 0 and 100 percentile. [#818](https://github.com/vaexio/vaex/pull/818)
      * Expression containing kwarg=True were treated as invalid. [#861](hhttps://github.com/vaexio/vaex/pull/861)
      * Unicode column names fully supported [#974](https://github.com/vaexio/vaex/issues/974)
   * Features
      * Datetime floor method [#843](https://github.com/vaexio/vaex/pull/843)
      * dropinf (similar to dropna) [#821](https://github.com/vaexio/vaex/pull/821)
      * Support for streaming from Google Cloud Storage. [#898](https://github.com/vaexio/vaex/pull/898)
      * IPython autocomplete support (e.g. `df['hom' (tab)`) [#961](https://github.com/vaexio/vaex/pull/961)
      * Out of core Parquet support using Arrow Dataset scanning [#993](https://github.com/vaexio/vaex/pull/993)
   * Refactor
      * Use `arrow.compute` for several string functions/kernels. [#885](https://github.com/vaexio/vaex/pull/885)
      * Separate DataFrame and Dataset. [#865](https://github.com/vaexio/vaex/pull/865)
   * Performance
      * concat (vaex.concat or df.concat) is about 100x faster. [#994](https://github.com/vaexio/vaex/pull/994)

## vaex-distributed (DEPRECATED)
   This is now part of vaex-enterprise (was a proof of content, never functional).

## vaex-graphql 0.2.0
  * Requirement changed to vaex-core >=4,<5

## vaex-hdf5 0.7.0
   * Requirement changed vaex-core >=4,<5

## vaex-jupyter 0.6.0
  * Requirement changed to vaex-core >=4,<5

## vaex-ml 0.11.0
   * Features
      * Batch training for CatBoost. [#819](https://github.com/vaexio/vaex/pull/819)
      * Support for `predict_proba` and `predict_log_proba` for sklearn classifiers. [#927](https://github.com/vaexio/vaex/pull/927)

## vaex-server 0.4.0
  * Requirement changed to vaex-core >=4,<5

## vaex-viz 0.5.0
  * Requirement changed to vaex-core >=4,<5

# vaex 3.1.0

## vaex-jupyter 0.5.2 (2020-6-12)
   * Features
      * Normalize histogram and change selection mode. [#826](https://github.com/vaexio/vaex/pull/826)

## vaex-ml 0.11.0-dev0
    * Features
      * Autogenerate the fast (or functional) API [#512](https://github.com/vaexio/vaex/pull/512)

## vaex-core 2.0.3 (2020-6-10)
   * Performance
      * isin uses hashmaps, leading to a 2x-4x performance increase for primitives, 200x for strings in some cases [#822](https://github.com/vaexio/vaex/pull/822)

## vaex-jupyter 0.5.1 (2020-6-4)
   * Features
      * Selection toggle list. [#797](https://github.com/vaexio/vaex/pull/797)

## vaex-server 0.3.1 (2020-6-4)
   * Fixes
      * Remote dataframe was still using dtype, not data_type. [#797](https://github.com/vaexio/vaex/pull/797)

## vaex-ml 0.10.0 (2020-6-4)
   * Features
      * Implementation of `GroupbyTransformer` [#479](https://github.com/vaexio/vaex/pull/479)

## vaex-arrow 0.6.1 (2020-6-4)
   * Fixes
      * Various fixes for aliased columns (column names with invalid identifiers) [#768](https://github.com/vaexio/vaex/pull/768)

## vaex-hdf5 0.6.1 (2020-6-4)
   * Fixes
      * Masked arrays supported in hdf5 files on s3 [#781](https://github.com/vaexio/vaex/pull/781)
      * Various fixes for aliased columns (column names with invalid identifiers) [#768](https://github.com/vaexio/vaex/pull/768)

## vaex-core 2.0.2 (2020-6-4)
   * Fixes
      * Masked arrays supported in hdf5 files on s3 [#781](https://github.com/vaexio/vaex/pull/781)
      * Expression.map always uses masked arrays to be state transferrable (a new dataset might have missing values) [#479](https://github.com/vaexio/vaex/pull/479)
      * Support importing Pandas dataframes with version 0.23 [#794](https://github.com/vaexio/vaex/pull/794)
      * Various fixes for aliased columns (column names with invalid identifiers) [#768](https://github.com/vaexio/vaex/pull/768) [#793](https://github.com/vaexio/vaex/pull/793)

## vaex-core 2.0.1 (2020-5-28)
   * Fixes
      * Join could in rare cases point to row 0, when there were values in the left, not present in the right [#765](https://github.com/vaexio/vaex/pull/765)
      * Tabulate 0.8.7 escaped html, undo this to print dataframes nicely.


# vaex 3.0.0 (2020-5-24)
   * Breaking changes:
     * Python 2 is not supported anymore
     * Variables don't have access to pi and e anymore
     * `df.rename_column` is now `df.rename` (and also renames variables)
     * DataFrame uses a normal dict instead of OrderedDict, requiring Python >= 3.6
     * Default limits (e.g. for plots) is minmax, so we don't miss outliers
     * `df.get_column_names()` returns the aliased names (invalid identifiers), pass `alias=False` to get the internal column name
     * Default value of `virtual` is True in method `df.export`, `df.to_dict`, `df.to_items`, `df.to_arrays`.
     * df.dtype is a property, to get data types for expressions, use df.data_type(), df.expr.dtype is still behaving the same
     * df.categorize takes min_value and max_value, and no longer needs the check argument, also the labels do not have to be strings.
     * vaex.open/from_csv etc does not copy the pandas index by default [#756](https://github.com/vaexio/vaex/pull/756)
     * df.categorize takes an inplace argument, similar to most methods, and returns the dataframe affected.


# vaex-core 2.0.0 (2020-5-24)
   * Performance
       * Printing out of dataframes done in 1 evaluate call, making remote dataframe printing faster. [#571](https://github.com/vaexio/vaex/pull/557)
       * Joining is faster and uses less memory (2x speedup measured) [#586](https://github.com/vaexio/vaex/pull/586)
       * Faster typechecks when adding columns of dtype=object (as often happens when coming from pandas) [#612](https://github.com/vaexio/vaex/pull/612)
       * Groupby 2x to 4x faster [#730](https://github.com/vaexio/vaex/pull/730)
   * Refactor
      * Task system is refactored, with task execution on CPU being default, and makes (de)serialization easier. [#571](https://github.com/vaexio/vaex/pull/557)
      * Serialization/encoding of data structures is more flexible, allowing binary blobs and json over the wire. [#571](https://github.com/vaexio/vaex/pull/557)
      * Execution and tasks support async await [#654](https://github.com/vaexio/vaex/pull/654)
   * Fixes
      * Renaming columns fixes [#571](https://github.com/vaexio/vaex/pull/571)
      * Joining with virtual columns but different data, and name collision fixes [#570](https://github.com/vaexio/vaex/pull/570)
      * Variables are treated similarly as columns, and respected in join [#573](https://github.com/vaexio/vaex/pull/573)
      * Arguments to lazy function which are numpy arrays gets put in the variables [#573](https://github.com/vaexio/vaex/pull/573)
      * Executor does not block after failed/interrupted tasks. [#571](https://github.com/vaexio/vaex/pull/557)
      * Default limits (e.g. for plots) is minmax, so we don't miss outliers [#581](https://github.com/vaexio/vaex/pull/581)
      * Do no fail printing out dataframe with 0 rows [#582](https://github.com/vaexio/vaex/pull/582)
      * Give proper NameError when using non-existing column names [#299](https://github.com/vaexio/vaex/pull/299)
      * Several fixes for concatenated dataframes.  [#590](https://github.com/vaexio/vaex/pull/590)
      * dropna/nan/missing only dropped rows when all column values were missing, if no columns were specified. [#600](https://github.com/vaexio/vaex/pull/600)
      * Flaky test for RobustScaler skipped for p36 [#614](https://github.com/vaexio/vaex/pull/614)
      * Copying/printing sparse matrices [#615](https://github.com/vaexio/vaex/pull/615)
      * Sparse columns names with invalid identifiers are not rewritten. [#617](https://github.com/vaexio/vaex/pull/617)
      * Column names with invalid identifiers which are rewritten are shown when printing the dataframe. [#617](https://github.com/vaexio/vaex/pull/617)
      * Column name rewriting for invalid identifiers also works on virtual columns. [#617](https://github.com/vaexio/vaex/pull/617)
      * Fix the links to the example datasets. [#609](https://github.com/vaexio/vaex/pull/609)
      * Expression.isin supports dtype=object [#669](https://github.com/vaexio/vaex/pull/669)
      * Fix `colum_count`, now only counts hidden columns if explicitly specified [#593](https://github.com/vaexio/vaex/pull/593)
      * df.values respects masked arrays [#640](https://github.com/vaexio/vaex/pull/640)
      * Rewriting a virtual column and doing a state transfer does not lead to `ValueError: list.remove(x): x not in list` [#592](https://github.com/vaexio/vaex/pull/592)
      * `df.<stat>(limits=...)` will now respect the selection [#651](https://github.com/vaexio/vaex/pull/651)
      * Using automatic names for aggregators led to many underscores in name [#687](https://github.com/vaexio/vaex/pull/687)
      * Support Python3.8 [#559](https://github.com/vaexio/vaex/pull/559)

   * Features
      * New lazy numpy wrappers: np.digitize and np.searchsorted [#573](https://github.com/vaexio/vaex/pull/573)
      * `df.to_arrow_table`/`to_pandas_df`/`to_items`/`df.to_dict`/`df.to_arrays` now take a chunk_size argument for chunked iterators [#589](https://github.com/vaexio/vaex/pull/589) (https://github.com/vaexio/vaex/pull/699)
      * Filtered datasets can be concatenated. [#590](https://github.com/vaexio/vaex/pull/590)
      * DataFrames/Executors are thread safe (meaning you can schedule/compute from any thread), which makes it work out of the box for Dash and Flask [#670](https://github.com/vaexio/vaex/pull/670)
      * `df.count/mean/std` etc can output in xarray.DataArray array type, makes plotting easier [#671](https://github.com/vaexio/vaex/pull/671)
      * Column names can have unicode, and we use str.isidentifier to test, also dont accidently hide columns. [#617](https://github.com/vaexio/vaex/pull/617)
      * Percentile approx can take a sequence of percentages [#527](https://github.com/vaexio/vaex/pull/527)
      * Polygon testing, useful in combinations with geo/geojson data [#685](https://github.com/vaexio/vaex/pull/685)
      * Added dt.quarter property and dt.strftime method to expression (by Juho Lauri) [#682](https://github.com/vaexio/vaex/pull/682)

# vaex-server 0.3.0 (2020-5-24)
   * Refactored server, can return multiple binary blobs, execute multiple tasks, cancel tasks, encoding/serialization is more flexible (like returning masked arrays). [#571](https://github.com/vaexio/vaex/pull/557)

# vaex-viz 0.4.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3

# vaex-graphql 0.1.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3

# vaex-astro 0.7.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3

# vaex-hdf5 0.6.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3

# vaex-ml 0.9.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3

# vaex-arrow 0.5.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3
   * Fixes
      * Booleans were negated, and didn't respect offsets.

# vaex-jupyter 0.5.0 (2020-5-24)
   * Requirement of vaex-core >=2,<3
   * Breaking changes
     * vaex-jupyter is refactored [#654](https://github.com/vaexio/vaex/pull/654)


# vaex 2.6.0 (2020-1-21)

# vaex-core 1.5.0
   * Features
      * df.evaluate_iterator for efficient parallel chunked evaluation [#515](https://github.com/vaexio/vaex/pull/515)
      * Widget progress bar has time estimation [#545](https://github.com/vaexio/vaex/pull/545)
   * Fixes
     * Slicing arrow string arrays with masked arrays is respected/working [#530](https://github.com/vaexio/vaex/pull/530)]

# vaex-ml 0.8.0
   * Performance
      * IncrementalPredictor uses parallel chunked support (2x speedup possible) [#515](https://github.com/vaexio/vaex/pull/515)
   * Fix
      * IncrementalPredictor: epochs now iterate over the whole DataFrame instead on a batch level [#523](https://github.com/vaexio/vaex/pull/523)
      * Rename `vaex.ml.sklearn.SKLearnPredictor` to `vaex.ml.sklearn.Predictor` [#524](https://github.com/vaexio/vaex/pull/524)
      * IncrementalPredictor can be used with `sklearn.linear_model.SGDClassifier` [539](https://github.com/vaexio/vaex/pull/539)
   * Features
      * CycleTransformer [#532](https://github.com/vaexio/vaex/pull/532)
      * BayesianTargetEncoder [#533](https://github.com/vaexio/vaex/pull/533)
      * WeightOfEvidenceEncoder [#534](https://github.com/vaexio/vaex/pull/534)
      * Improve the consistency of the vaex.ml API for model transformers [#552](https://github.com/vaexio/vaex/pull/552)

# vaex 2.5.0 (2019-12-16)

# vaex-core 1.4.0
   * Performance
      * Dataframes are always true (implements `__bool__`) to avoid calling `__len__` [#496](https://github.com/vaexio/vaex/pull/496)
   * Fixes
      * Do not duplicate column when joining DataFrames on a column with the same name [#480](https://github.com/vaexio/vaex/pull/480)
      * Better error messages/stack traces, and work better with debugger. [#488](https://github.com/vaexio/vaex/pull/488)
      * Accept numpy scalars in expressions. [#462](https://github.com/vaexio/vaex/pull/462)
      * Expression.astype can create datetime64 columns out of (arrow) strings arrays. [#440](https://github.com/vaexio/vaex/pull/440)
      * Invalid mask access triggered when memory-mapped read only for strings. [#459](https://github.com/vaexio/vaex/pull/459)
   * Features
      * Expressions are not evaluated for filtered data [#483](https://github.com/vaexio/vaex/pull/483) [#496](https://github.com/vaexio/vaex/pull/496) and selections [505](https://github.com/vaexio/vaex/pull/505)
      * Filtering (using df.filter) allows more flexible (and growing/expanding!) filter. [#489](https://github.com/vaexio/vaex/pull/489)
      * Filtering and selections allow for booleans (True or False) to auto 'broadcast', to allow 'conditional filtering'. [#489](https://github.com/vaexio/vaex/pull/489)

# vaex-ml 0.7.0
   * Features
      * IncrementalPredictor for `scikit-learn` models that support the `.partial_fit` method [#497](https://github.com/vaexio/vaex/pull/497)
   * Fixes
      * Adding unique function names to dataframes to enable adding a predictor twice [#492](https://github.com/vaexio/vaex/pull/492)

## vaex-arrow 0.4.2
      * Compatibility with vaex-core 1.4.0

# vaex 2.4.0 (2019-11-26)

## vaex-core 1.3.0

   * Performance
      * Parallel df.evaluate [#474](https://github.com/vaexio/vaex/pull/474)
      * Avoid calling df.get_column_names (1000x for 1 billion rows per column use) [#473](https://github.com/vaexio/vaex/pull/473)
      * Slicing e.g df[1:-1] goes much faster for filtered dataframes [#471](https://github.com/vaexio/vaex/pull/471)
      * Dataframe copying and expression rewriting was slow [#470](https://github.com/vaexio/vaex/pull/470)
      * Double indices columns were not using index cache since empty dict is falsy [#439](https://github.com/vaexio/vaex/pull/439)
   * Features
      * multi-key sorting of a DataFrame [#463](https://github.com/vaexio/vaex/pull/463)
      * vaex expression to pandas.Series support [#456](https://github.com/vaexio/vaex/pull/456)
      * Dask array support [#449](https://github.com/vaexio/vaex/pull/449) [#476](https://github.com/vaexio/vaex/pull/476) [example](http://docs.vaex.io/en/latest/example_dask.html)
      * isin() method for expressions [#441](https://github.com/vaexio/vaex/pull/441) [docs](api.html#vaex.expression.Expression.isin)
      * Existing expressions are rewritten, to make them behave like arrays [#450](https://github.com/vaexio/vaex/pull/450)

## vaex-hdf5 0.5.6
   * requires vaex-core >=1.3,<2 for parallel evaluate

## vaex-jupyter 0.4.1
   * Fixes:
      * bqplot 0.12 revealed a bug/inconsistency with heatmap [#465](https://github.com/vaexio/vaex/pull/465)

## vaex-arrow 0.4.1
   * Fixes
      *  Support for Apache Arrow >= 0.15

## vaex-ml 0.6.2
   * Fixes
      * Docstrings and minor improvements

## vaex-graphql 0.0.1 (2019-10-15)
   * initial release 0.1

# vaex 2.3.0 (2019-10-15)

## vaex-core 1.2.0
   * feature: auto upcasting for sum [#435](https://github.com/vaexio/vaex/pull/435)
   * fix: selection/filtering fix when using masked values [#431](https://github.com/vaexio/vaex/pull/431)
   * fix: masked string array fixes [#434](https://github.com/vaexio/vaex/pull/434)
   * fix: memory usage fix for joins [#439](https://github.com/vaexio/vaex/pull/439)

## vaex-arrow 0.4.1
 * fix: support for Apache Arrow >= 0.15

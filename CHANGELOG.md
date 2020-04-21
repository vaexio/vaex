# vaex 3.0.0-dev (unreleased)
   * Breaking changes:
     * Python 2 is not supported anymore
     * Variables don't have access to pi and e anymore
     * df.rename_column is now df.rename (and also renames variables)
     * DataFrame uses a normal dict instead of OrderedDict, requiring Python >= 3.6
     * Default limits (e.g. for plots) is minmax, so we don't miss outliers
     * df.get_column_names() returns the aliased names (invalid identifiers), pass alias=False to get the internal column name

# vaex-core 2.0.0-dev
   * Performance
       * Printing out of dataframes done in 1 evaluate call, making remote dataframe printing faster. [#571](https://github.com/vaexio/vaex/pull/557)
       * Joining is faster and uses less memory (2x speedup measured) [#586](https://github.com/vaexio/vaex/pull/586)
       * Faster typechecks when adding columns of dtype=object (as often happens when coming from pandas) [#612](https://github.com/vaexio/vaex/pull/612)
   * Refactor
      * Task system is refactored, with task execution on CPU being default, and makes (de)serialization easier. [#571](https://github.com/vaexio/vaex/pull/557)
      * Serialization/encoding of data structures is more flexible, allowing binary blobs and json over the wire. [#571](https://github.com/vaexio/vaex/pull/557)
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
      * Fix `colum_count`, now only counts hidden columns if expicitly specified [#593](https://github.com/vaexio/vaex/pull/593)
      * df.values respects masked arrays [#640](https://github.com/vaexio/vaex/pull/640)
      * Rewriting a virtual column and doing a state transfer does not lead to `ValueError: list.remove(x): x not in list` [#592](https://github.com/vaexio/vaex/pull/592)
      * `df.<stat>(limits=...)` will now respect the selection [#651](https://github.com/vaexio/vaex/pull/651)
      * Using automatic names for aggregators led to many underscores in name [#687](https://github.com/vaexio/vaex/pull/687)
   * Features
      * New lazy numpy wrappers: np.digitize and np.searchsorted [#573](https://github.com/vaexio/vaex/pull/573)
      * df.to_arrow_table/to_pandas_df/to_items now take a chunk_size argument for chunked iterators [#589](https://github.com/vaexio/vaex/pull/589)
      * Filtered datasets can be concatenated. [#590](https://github.com/vaexio/vaex/pull/590)
      * DataFrames/Executors are thread safe (meaning you can schedule/compute from any thread), which makes it work out of the box for Dash and Flask [#670](https://github.com/vaexio/vaex/pull/670)
      * df.count/mean/std etc can output in xarray.DataArray array type, makes plotting easier [#671](https://github.com/vaexio/vaex/pull/671)
      * Column names can have unicode, and we use str.isidentifier to test, also dont accidently hide columns. [#617](https://github.com/vaexio/vaex/pull/617)
      * Percentile approx can take a sequence of percentages [#527](https://github.com/vaexio/vaex/pull/527)
      * Polygon testing, useful in combinations with geo/geojson data [#685](https://github.com/vaexio/vaex/pull/685)

# vaex-server 0.3.0-dev
   * Refactored server, can return multiple binary blobs, execute multiple tasks, cancel tasks, encoding/serialization is more flexible (like returning masked arrays). [#571](https://github.com/vaexio/vaex/pull/557)

# vaex-viz 0.4.0-dev
   * Requirement of vaex-core >=2,<3

# vaex-graphql 0.1.0-dev
   * Requirement of vaex-core >=2,<3

# vaex-astro 0.7.0-dev
   * Requirement of vaex-core >=2,<3

# vaex-hdf5 0.6.0-dev
   * Requirement of vaex-core >=2,<3

# vaex-ml 0.9.0-dev
   * Requirement of vaex-core >=2,<3

# vaex-arrow 0.5.0-dev
   * Requirement of vaex-core >=2,<3

# vaex-jupyter 0.5.0-dev
   * Requirement of vaex-core >=2,<3


# vaex 2.6.0 (2020-1-21)

# vaex-core 1.5.0
   * Features
      * df.evalute_iterator for efficient parallel chunked evaluation [#515](https://github.com/vaexio/vaex/pull/515)
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

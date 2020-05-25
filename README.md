
[![Documentation](https://readthedocs.org/projects/vaex/badge/?version=latest)](https://docs.vaex.io)

# What is Vaex?

Vaex is a Python library for lazy **Out-of-Core DataFrames** (similar to
Pandas), to visualize and explore big tabular datasets. It can calculate
*statistics* such as mean, sum, count, standard deviation etc, on an
*N-dimensional grid* for more than **a billion** (`10^9`) objects/rows
**per second**. Visualization is done using **histograms**, **density
plots** and **3d volume rendering**, allowing interactive exploration of
big data. Vaex uses memory mapping, zero memory copy policy and lazy
computations for best performance (no memory wasted).

# Key features
## Instant opening of Huge data files (memory mapping) ⚡
Hdf5 and [Apache Arrow](https://arrow.apache.org/) supported. 


## Expression system 🧪
Don't waste memory or time with feature engineering, we (lazily) transform your data when needed.

## Out of core DataFrame
Filtering and computing will be waste memory by making copies, the data is kept untouched on disk, and will be streamed over when needed.

## Fast groupby / aggregations
Parallel groupby, *very* fast, especially when using categories (>1 billion/second).

## Fast and efficient join
We don't copy/materialize the 'right' table when joining, saving gigabytes of memory. With subsecond joining on a billion rows, it's pretty fast!

## More features

 * Remote dataframes (documentation coming soon)
 * Integration into [Jupyter and Voila for interactive notebooks and dashboards](https://vaex.readthedocs.io/en/latest/tutorial_jupyter.html)
 * [Machine Learning without (explicit) pipelines](https://vaex.readthedocs.io/en/latest/tutorial_ml.html) 
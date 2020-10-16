
[![Documentation](https://readthedocs.org/projects/vaex/badge/?version=latest)](https://docs.vaex.io)

# What is Vaex?

Vaex is a high performance Python library for lazy **Out-of-Core DataFrames**
(similar to Pandas), to visualize and explore big tabular datasets. It
calculates *statistics* such as mean, sum, count, standard deviation etc, on an
*N-dimensional grid* for more than **a billion** (`10^9`) samples/rows **per
second**. Visualization is done using **histograms**, **density plots** and **3d
volume rendering**, allowing interactive exploration of big data. Vaex uses
memory mapping, zero memory copy policy and lazy computations for best
performance (no memory wasted).

# Installing
With pip:
```
$ pip install vaex
```
Or conda:
```
$ conda install -c conda-forge vaex
```

[For more details, see the documentation](https://docs.vaex.io/en/latest/installing.html)

# Key features
## Instant opening of Huge data files (memory mapping)
[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) and [Apache Arrow](https://arrow.apache.org/) supported.

![opening1a](https://user-images.githubusercontent.com/1765949/82818563-31c1e200-9e9f-11ea-9ee0-0a8c1994cdc9.png)


![opening1b](https://user-images.githubusercontent.com/1765949/82820352-49e73080-9ea2-11ea-9153-d73aa399d329.png)

[Read the documentation on how to efficiently convert your data](https://docs.vaex.io/en/latest/example_io.html) from CSV files, Pandas DataFrames, or other sources.


Lazy streaming from S3 supported in combination with memory mapping.

![opening1c](https://user-images.githubusercontent.com/1765949/82820516-a21e3280-9ea2-11ea-948b-07df26c4b5d3.png)


## Expression system
Don't waste memory or time with feature engineering, we (lazily) transform your data when needed.


![expression](https://user-images.githubusercontent.com/1765949/82818733-70f03300-9e9f-11ea-80b0-ab28e7950b5c.png)



## Out-of-core DataFrame
Filtering and evaluating expressions will not waste memory by making copies; the data is kept untouched on disk, and will be streamed only when needed. Delay the time before you need a cluster.


![occ-animated](https://user-images.githubusercontent.com/1765949/82821111-c6c6da00-9ea3-11ea-9f9e-498de8133cc2.gif)

## Fast groupby / aggregations
Vaex implements parallelized, highly performant `groupby` operations, especially when using categories (>1 billion/second).


![groupby](https://user-images.githubusercontent.com/1765949/82818807-97ae6980-9e9f-11ea-8820-41dd4441057a.png)


## Fast and efficient join
Vaex doesn't copy/materialize the 'right' table when joining, saving gigabytes of memory. With subsecond joining on a billion rows, it's pretty fast!

![join](https://user-images.githubusercontent.com/1765949/82818840-a268fe80-9e9f-11ea-8ba2-6a6d52c4af88.png)

## More features

 * Remote DataFrames (documentation coming soon)
 * Integration into [Jupyter and Voila for interactive notebooks and dashboards](https://vaex.readthedocs.io/en/latest/tutorial_jupyter.html)
 * [Machine Learning without (explicit) pipelines](https://vaex.readthedocs.io/en/latest/tutorial_ml.html)


# Learn more about Vaex
 * Articles
   * [Beyond Pandas: Spark, Dask, Vaex and other big data technologies battling head to head](https://towardsdatascience.com/beyond-pandas-spark-dask-vaex-and-other-big-data-technologies-battling-head-to-head-a453a1f8cc13) (includes benchmarks)
   * [7 reasons why I love Vaex for data science](https://towardsdatascience.com/7-reasons-why-i-love-vaex-for-data-science-99008bc8044b) (tips and trics)
   * [ML impossible: Train 1 billion samples in 5 minutes on your laptop using Vaex and Scikit-Learn](https://towardsdatascience.com/ml-impossible-train-a-1-billion-sample-model-in-20-minutes-with-vaex-and-scikit-learn-on-your-9e2968e6f385)
   * [How to analyse 100 GB of data on your laptop with Python](https://towardsdatascience.com/how-to-analyse-100s-of-gbs-of-data-on-your-laptop-with-python-f83363dda94)
   * [Flying high with Vaex: analysis of over 30 years of flight data in Python](https://towardsdatascience.com/https-medium-com-jovan-veljanoski-flying-high-with-vaex-analysis-of-over-30-years-of-flight-data-in-python-b224825a6d56)
   * [Vaex: A DataFrame with super strings - Speed up your text processing up to a 1000x
](https://towardsdatascience.com/vaex-a-dataframe-with-super-strings-789b92e8d861)
   * [Vaex: Out of Core Dataframes for Python and Fast Visualization - 1 billion row datasets on your laptop](https://towardsdatascience.com/vaex-out-of-core-dataframes-for-python-and-fast-visualization-12c102db044a)

 * [Follow our tutorials](https://docs.vaex.io/en/latest/tutorials.html)
 * Watch our more recent talks:
   * [PyData London 2019](https://www.youtube.com/watch?v=2Tt0i823-ec)
   * [SciPy 2019](https://www.youtube.com/watch?v=ELtjRdPT8is)
 * Contact us for data science solutions, training, or enterprise support at https://vaex.io/

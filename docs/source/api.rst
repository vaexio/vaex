API documentation for vaex library
==================================


Quick lists
-----------
Opening/reading in your data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    vaex.open
    vaex.open_many
    vaex.from_arrays
    vaex.from_arrow_dataset
    vaex.from_arrow_table
    vaex.from_ascii
    vaex.from_astropy_table
    vaex.from_csv
    vaex.from_csv_arrow
    vaex.from_dataset
    vaex.from_dict
    vaex.from_items
    vaex.from_json
    vaex.from_pandas
    vaex.from_records

Visualizations.
~~~~~~~~~~~~~~~

.. autosummary::

     vaex.viz.DataFrameAccessorViz.heatmap
     vaex.viz.DataFrameAccessorViz.histogram
     vaex.viz.DataFrameAccessorViz.scatter

Statistics.
~~~~~~~~~~~

.. autosummary::

    vaex.dataframe.DataFrame.correlation
    vaex.dataframe.DataFrame.count
    vaex.dataframe.DataFrame.cov
    vaex.dataframe.DataFrame.max
    vaex.dataframe.DataFrame.mean
    vaex.dataframe.DataFrame.median_approx
    vaex.dataframe.DataFrame.min
    vaex.dataframe.DataFrame.minmax
    vaex.dataframe.DataFrame.mode
    vaex.dataframe.DataFrame.mutual_information
    vaex.dataframe.DataFrame.std
    vaex.dataframe.DataFrame.unique
    vaex.dataframe.DataFrame.var


.. toctree::

vaex-core
---------

.. automodule:: vaex
    :members: concat, delayed, example, from_arrays, from_arrow_dataset, from_arrow_table, from_ascii, from_astropy_table, from_csv, from_dataset, from_dict, from_items, from_json, from_pandas, from_records, open, open_many, register_function, server, vconstant, vrange
    :undoc-members:
    :show-inheritance:

Aggregation and statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vaex.stat
    :members:


.. automodule:: vaex.agg
    :members:

.. .. autoclass:: vaex.stat.Statistic
..     :members:

Caching
~~~~~~~

.. automodule:: vaex.cache
    :members: is_on, off, get, set, memory, memory_infinite, disk, redis
    :undoc-members:
    :show-inheritance:

DataFrame class
~~~~~~~~~~~~~~~

.. autoclass:: vaex.dataframe.DataFrame
     :members:
     :special-members:


DataFrameLocal class
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.dataframe.DataFrameLocal
     :members:
     :special-members:

Date/time operations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.DateTime
     :members:
     :special-members:

Expression class
~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.Expression
     :members:
     :special-members:

Geo operations
~~~~~~~~~~~~~~

.. autoclass:: vaex.geo.DataFrameAccessorGeo
     :members:
     :special-members:

Logging
~~~~~~~

.. automodule:: vaex.logging
    :members:

String operations
~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.StringOperations
     :members:
     :special-members:

String (pandas) operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.StringOperationsPandas
     :members:
     :special-members:

Struct (arrow) operations
~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.StructOperations
     :members:
     :special-members:

Timedelta operations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.TimeDelta
     :members:
     :special-members:


vaex-graphql
------------------

.. autoclass:: vaex.graphql.DataFrameAccessorGraphQL
     :members:
     :special-members:

vaex-jupyter
------------

.. autoclass:: vaex.jupyter.DataFrameAccessorWidget
     :members:
     :special-members:

.. automodule:: vaex.jupyter
    :members: debounced, flush, interactive_selection
    :undoc-members:
    :show-inheritance:

vaex.jupyter.model
~~~~~~~~~~~~~~~~~~

.. automodule:: vaex.jupyter.model
    :members:
    :undoc-members:
    :show-inheritance:

vaex.jupyter.view
~~~~~~~~~~~~~~~~~~

.. automodule:: vaex.jupyter.view
    :members:
    :undoc-members:
    :show-inheritance:

vaex.jupyter.widgets
~~~~~~~~~~~~~~~~~~~~

.. automodule:: vaex.jupyter.widgets
    :members:
    :undoc-members:
    :show-inheritance:


vaex-ml
-------

See the `ML tutorial <tutorial_ml.ipynb>`_ an introduction, and the `ML examples <examples.rst>`_ for more advanced usage.


Transformers & Encoders
~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

    vaex.ml.transformations.FrequencyEncoder
    vaex.ml.transformations.LabelEncoder
    vaex.ml.transformations.MaxAbsScaler
    vaex.ml.transformations.MinMaxScaler
    vaex.ml.transformations.OneHotEncoder
    vaex.ml.transformations.MultiHotEncoder
    vaex.ml.transformations.PCA
    vaex.ml.transformations.RobustScaler
    vaex.ml.transformations.StandardScaler
    vaex.ml.transformations.CycleTransformer
    vaex.ml.transformations.BayesianTargetEncoder
    vaex.ml.transformations.WeightOfEvidenceEncoder
    vaex.ml.transformations.KBinsDiscretizer
    vaex.ml.transformations.GroupByTransformer


.. autoclass:: vaex.ml.transformations.FrequencyEncoder
     :members:

.. autoclass:: vaex.ml.transformations.LabelEncoder
     :members:

.. autoclass:: vaex.ml.transformations.MaxAbsScaler
     :members:

.. autoclass:: vaex.ml.transformations.MinMaxScaler
     :members:

.. autoclass:: vaex.ml.transformations.OneHotEncoder
     :members:

.. autoclass:: vaex.ml.transformations.MultiHotEncoder
     :members:

.. autoclass:: vaex.ml.transformations.PCA
     :members:

.. autoclass:: vaex.ml.transformations.RobustScaler
     :members:

.. autoclass:: vaex.ml.transformations.StandardScaler
     :members:

.. autoclass:: vaex.ml.transformations.CycleTransformer
     :members:

.. autoclass:: vaex.ml.transformations.BayesianTargetEncoder
     :members:

.. autoclass:: vaex.ml.transformations.WeightOfEvidenceEncoder
     :members:

.. autoclass:: vaex.ml.transformations.KBinsDiscretizer
     :members:

.. autoclass:: vaex.ml.transformations.GroupByTransformer
     :members:


Clustering
~~~~~~~~~~

.. autosummary::

    vaex.ml.cluster.KMeans

.. autoclass:: vaex.ml.cluster.KMeans
     :members:


Metrics
~~~~~~~


.. autoclass:: vaex.ml.metrics.DataFrameAccessorMetrics
     :members:


Scikit-learn
~~~~~~~~~~~~

.. autosummary::

    vaex.ml.sklearn.IncrementalPredictor
    vaex.ml.sklearn.Predictor

.. automodule:: vaex.ml.sklearn
    :members:
    :undoc-members:
    :show-inheritance:


Boosted trees
~~~~~~~~~~~~~

.. autosummary::

    vaex.ml.lightgbm.LightGBMModel
    vaex.ml.xgboost.XGBoostModel
    vaex.ml.catboost.CatBoostModel


.. autoclass:: vaex.ml.lightgbm.LightGBMModel
     :members:

.. autoclass:: vaex.ml.xgboost.XGBoostModel
     :members:

.. autoclass:: vaex.ml.catboost.CatBoostModel
     :members:

Tensorflow
~~~~~~~~~~

.. autosummary::

     vaex.ml.tensorflow.KerasModel

.. autoclass:: vaex.ml.tensorflow.KerasModel
     :members:

.. autoclass:: vaex.ml.tensorflow.DataFrameAccessorTensorflow
     :members:


Incubator/experimental
~~~~~~~~~~~~~~~~~~~~~~

These models are in the incubator phase and may disappear in the future

.. .. autoclass:: vaex.ml.incubator.pygbm.PyGBMModel
..      :members:

.. autoclass:: vaex.ml.incubator.annoy.ANNOYModel
     :members:

.. autoclass:: vaex.ml.incubator.river.RiverModel
     :members:


vaex-viz
--------

.. autoclass:: vaex.viz.DataFrameAccessorViz
     :members:
     :special-members:

.. autoclass:: vaex.viz.ExpressionAccessorViz
     :members:
     :special-members:


.. .. Subpackages
.. .. -----------

.. .. toctree::

..     vaex.notebook


.. Submodules
.. ----------


.. vaex.dataframe module
.. -------------------

.. .. automodule:: vaex.dataframe
..     :members:
..     :undoc-members:
..     :show-inheritance:




.. vaex.dataframe module
.. -------------------

.. .. automodule:: vaex.dataframe
..     :members: Dataset, DatasetLocal, DatasetConcatenated, DatasetArrays, DatasetMemoryMapped
..     :undoc-members:
..     :show-inheritance:

.. vaex.events module
.. ------------------

.. .. automodule:: vaex.events
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.execution module
.. ---------------------

.. .. automodule:: vaex.execution
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.grids module
.. -----------------

.. .. automodule:: vaex.grids
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.kld module
.. ---------------

.. .. automodule:: vaex.kld
..     :members:
..     :undoc-members:
..     :show-inheritance:


.. vaex.multithreading module
.. --------------------------

.. .. automodule:: vaex.multithreading
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.quick module
.. -----------------

.. .. automodule:: vaex.quick
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.remote module
.. ------------------

.. .. automodule:: vaex.remote
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.samp module
.. ----------------

.. .. automodule:: vaex.samp
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.settings module
.. --------------------

.. .. automodule:: vaex.settings
..     :members:
..     :undoc-members:
..     :show-inheritance:

.. vaex.utils module
.. -----------------

.. .. automodule:: vaex.utils
..     :members:
..     :undoc-members:
..     :show-inheritance:





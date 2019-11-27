API documentation for vaex library
==================================


Quick lists
-----------
Opening/reading in your data.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
    
    vaex.open
    vaex.from_arrow_table
    vaex.from_arrays
    vaex.from_dict
    vaex.from_csv
    vaex.from_ascii
    vaex.from_pandas
    vaex.from_astropy_table

Visualization.
~~~~~~~~~~~~~~

.. autosummary::
    
    vaex.dataframe.DataFrame.plot
    vaex.dataframe.DataFrame.plot1d
    vaex.dataframe.DataFrame.scatter
    vaex.dataframe.DataFrame.plot_widget
    vaex.dataframe.DataFrame.healpix_plot


Statistics.
~~~~~~~~~~~

.. autosummary::
    
    vaex.dataframe.DataFrame.count
    vaex.dataframe.DataFrame.mean
    vaex.dataframe.DataFrame.std
    vaex.dataframe.DataFrame.var
    vaex.dataframe.DataFrame.cov
    vaex.dataframe.DataFrame.correlation
    vaex.dataframe.DataFrame.median_approx
    vaex.dataframe.DataFrame.mode
    vaex.dataframe.DataFrame.min
    vaex.dataframe.DataFrame.max
    vaex.dataframe.DataFrame.minmax
    vaex.dataframe.DataFrame.mutual_information


.. toctree::

vaex-core
---------

.. automodule:: vaex
    :members: open, from_arrays, from_dict, from_items, from_arrow_table, from_csv, from_ascii, from_pandas, from_astropy_table, from_samp, open_many, register_function, server, example, app, delayed
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


Expression class
~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.Expression
     :members:
     :special-members:

Aggregation and statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: vaex.stat
    :members:


.. automodule:: vaex.agg
    :members:

.. .. autoclass:: vaex.stat.Statistic
..     :members:


Extensions
----------

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

Date/time operations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.DateTime
     :members:
     :special-members:

Timedelta operations
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.expression.TimeDelta
     :members:
     :special-members:

Geo operations
~~~~~~~~~~~~~~

.. autoclass:: vaex.geo.DataFrameAccessorGeo
     :members:
     :special-members:

GraphQL operations
~~~~~~~~~~~~~~~~~~

.. autoclass:: vaex.graphql.DataFrameAccessorGraphQL
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




Machine learning with vaex.ml
-----------------------------


Clustering
~~~~~~~~~~

.. autoclass:: vaex.ml.cluster.KMeans
     :members:

PCA
~~~

.. autoclass:: vaex.ml.transformations.PCA
     :members:

Encoders
~~~~~~~~

.. autoclass:: vaex.ml.transformations.LabelEncoder
     :members:

.. autoclass:: vaex.ml.transformations.OneHotEncoder
     :members:

.. autoclass:: vaex.ml.transformations.StandardScaler
     :members:

.. autoclass:: vaex.ml.transformations.MinMaxScaler
     :members:

.. autoclass:: vaex.ml.transformations.MaxAbsScaler
     :members:

.. autoclass:: vaex.ml.transformations.RobustScaler
     :members:

Boosted trees
~~~~~~~~~~~~~

.. autoclass:: vaex.ml.lightgbm.LightGBMModel
     :members:

.. autoclass:: vaex.ml.xgboost.XGBoostModel
     :members:

.. PyGBM support is in the incubator phase, which means support may disappear in future versions

.. .. autoclass:: vaex.ml.incubator.pygbm.PyGBMModel
..      :members:

Nearest neighbour
~~~~~~~~~~~~~~~~~

Annoy support is in the incubator phase, which means support may disappear in future versions

.. autoclass:: vaex.ml.incubator.annoy.ANNOYModel
     :members:

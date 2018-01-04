API documentation for vaex library
==================================


Quick list for opening/reading in your data.
--------------------------------------------

.. autosummary::
    
    vaex.open
    vaex.from_arrays
    vaex.from_csv
    vaex.from_ascii
    vaex.from_pandas
    vaex.from_astropy_table

Quick list for visualization.
-----------------------------

.. autosummary::
    
    vaex.dataset.Dataset.plot
    vaex.dataset.Dataset.plot1d
    vaex.dataset.Dataset.scatter
    vaex.dataset.Dataset.plot_widget
    vaex.dataset.Dataset.healpix_plot


Quick list for statistics.
--------------------------

.. autosummary::
    
    vaex.dataset.Dataset.count
    vaex.dataset.Dataset.mean
    vaex.dataset.Dataset.std
    vaex.dataset.Dataset.var
    vaex.dataset.Dataset.cov
    vaex.dataset.Dataset.correlation
    vaex.dataset.Dataset.median_approx
    vaex.dataset.Dataset.mode
    vaex.dataset.Dataset.min
    vaex.dataset.Dataset.max
    vaex.dataset.Dataset.minmax
    vaex.dataset.Dataset.mutual_information


.. toctree::

vaex module
-----------

.. automodule:: vaex
    :members: open, from_arrays, from_csv, from_ascii, from_pandas, from_astropy_table, from_samp, open_many, server, example, app, zeldovich, set_log_level_debug, set_log_level_info, set_log_level_warning, set_log_level_exception, set_log_level_off, delayed
    :undoc-members:
    :show-inheritance:


Dataset class
-------------

.. autoclass:: vaex.dataset.Dataset
     :members:
     :special-members:


vaex.stat module
----------------

.. automodule:: vaex.stat
    :members:


.. autoclass:: vaex.stat.Statistic
    :members:


.. .. Subpackages
.. .. -----------

.. .. toctree::

..     vaex.notebook


.. Submodules
.. ----------


.. vaex.dataset module
.. -------------------

.. .. automodule:: vaex.dataset
..     :members:
..     :undoc-members:
..     :show-inheritance:




.. vaex.dataset module
.. -------------------

.. .. automodule:: vaex.dataset
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

Note that vaex.ml does not fall under the MIT, but the `CC BY-CC-ND <https://creativecommons.org/licenses/by-nc-nd/4.0/>`_ LICENSE, which means it's ok for personal or academic use. You can install vaex-ml using `pip install vaex-ml`.

.. autoclass:: vaex.ml.cluster.KMeans
     :members:
     :special-members:

.. autoclass:: vaex.ml.transformations.MinMaxScaler
     :members:
     :special-members:

.. autoclass:: vaex.ml.transformations.StandardScaler
     :members:
     :special-members:

.. autoclass:: vaex.ml.transformations.PCA
     :members:
     :special-members:

.. autoclass:: vaex.ml.xgboost.XGBoost
     :members:
     :special-members:


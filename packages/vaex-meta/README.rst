|Travis| |Conda| |Chat| 

Vaex uses several sites:

* Main page: https://vaex.io/
* Documentation: https://docs.vaex.io/
* Github: https://github.com/vaexio/vaex
* PyPi: https://pypi.python.org/pypi/vaex/


Vaex is open source software, if you need support, contact us at https://vaex.io



What is Vaex?
-------------

Vaex is a python library for lazy **Out-of-Core DataFrames** (similar to
Pandas), to visualize and explore big tabular datasets. It can calculate
*statistics* such as mean, sum, count, standard deviation etc, on an
*N-dimensional grid* for more than **a billion** (10^9) objects/rows
**per second**. Visualization is done using **histograms**, **density
plots** and **3d volume rendering**, allowing interactive exploration of
big data. Vaex uses memory mapping, zero memory copy policy and lazy
computations for best performance (no memory wasted).


Why vaex
========

-  **Performance:** Works with huge tabular data, process
   more than a *billion* rows/second
-  **Lazy / Virtual columns:** compute on the fly, without wasting ram
-  **Memory efficient** no memory copies when doing
   filtering/selections/subsets.
-  **Visualization:** directly supported, a one-liner is often enough.
-  **User friendly API:** You will only need to deal with a Dataset
   object, and tab completion + docstring will help you out:
   ``ds.mean<tab>``, feels very similar to Pandas.
-  **Lean:** separated into multiple packages

   -  ``vaex-core``: Dataset and core algorithms, takes numpy arrays as
      input columns.
   -  ``vaex-hdf5``: Provides memory mapped numpy arrays to a Dataset.
   -  ``vaex-arrow``: `Arrow <https://arrow.apache.org/>`__ support for
      cross language data sharing.
   -  ``vaex-viz``: Visualization based on matplotlib.
   -  ``vaex-jupyter``: Interactive visualization based on Jupyter
      widgets / ipywidgets, bqplot, ipyvolume and ipyleaflet.
   -  ``vaex-astro``: Astronomy related transformations and FITS file
      support.
   -  ``vaex-server``: Provides a server to access a dataset remotely.
   -  ``vaex-distributed``: (Proof of concept) combined multiple servers
      / cluster into a single dataset for distributed computations.
   -  ``vaex-qt``: Program written using Qt GUI.
   -  ``vaex``: meta package that installs all of the above.
   -  ``vaex-ml``: `Machine learning <http://docs.vaex.io/en/latest/ml.html>`__ with automatic pipelines.

-  **Jupyter integration**: vaex-jupyter will give you interactive
   visualization and selection in the Jupyter notebook and Jupyter lab.

Installation
------------

Using conda:

-  ``conda install -c conda-forge vaex``

Using pip:

-  ``pip install vaex``

Or read the `detailed instructions <https://docs.vaex.io/en/latest/installing.html>`__

Getting started
===============

We assuming you have installed vaex, and are running a `Jupyter notebook
server <https://jupyter.readthedocs.io/en/latest/running.html>`__. We
start by importing vaex and ask it to give us sample example dataset.

.. code:: ipython3

    import vaex
    ds = vaex.example()  # open the example dataset provided with vaex


Instead, you can `download some larger datasets <https://docs.vaex.io/en/latest/datasets.html>`__, or
`read in your csv file <https://docs.vaex.io/en/latest/api.html#vaex.from_csv>`__.

.. code:: ipython3

    ds  # will pretty print a table





Using `square brackets[] <https://docs.vaex.io/en/latest/api.html#vaex.dataset.Dataset.__getitem__>`__,
we can easily filter or get different views on the dataset.

.. code:: ipython3

    ds_negative = ds[ds.x < 0]  # easily filter your dataset, without making a copy
    ds_negative[:5][['x', 'y']]  # take the first five rows, and only the 'x' and 'y' column (no memory copy!)






When dealing with huge datasets, say a billion rows (10^9),
computations with the data can waste memory, up to 8 GB for a new
column. Instead, vaex uses lazy computation, only a representation of
the computation is stored, and computations done on the fly when needed.
Even though, you can just many of the numpy functions, as if it was a
normal array.

.. code:: ipython3

    import numpy as np
    # creates an expression (nothing is computed)
    r = np.sqrt(ds.x**2 + ds.y**2 + ds.z**2)
    r  # for convenience, we print out some values




.. parsed-literal::

    <vaex.expression.Expression(expressions='sqrt((((x ** 2) + (y ** 2)) + (z ** 2)))')> instance at 0x11bcc4780 values=[2.9655450396553587, 5.77829281049018, 6.99079603950256, 9.431842752707537, 0.8825613121347967 ... (total 330000 values) ... 7.453831761514681, 15.398412491068198, 8.864250273925633, 17.601047186042507, 14.540181524970293] 



These expressions can be added to the dataset, creating what we call a
*virtual column*. These virtual columns are simular to normal columns,
except they do not waste memory.

.. code:: ipython3

    ds['r'] = r  # add a (virtual) column that will be computed on the fly
    ds.mean(ds.x), ds.mean(ds.r)  # calculate statistics on normal and virtual columns




.. parsed-literal::

    (-0.06713149126400597, 9.407082338299773)



One of the core features of vaex is its ability to calculate statistics
on a regular (N-dimensional) grid. The dimensions of the grid are
specified by the binby argument (analogous to SQL's grouby), and the
shape and limits.

.. code:: ipython3

    ds.mean(ds.r, binby=ds.x, shape=32, limits=[-10, 10]) # create statistics on a regular grid (1d)




.. parsed-literal::

    array([15.01058183, 14.43693006, 13.72923338, 12.90294499, 11.86615103,
           11.03563695, 10.12162553,  9.2969267 ,  8.58250973,  7.86602644,
            7.19568442,  6.55738773,  6.01942499,  5.51462457,  5.15798991,
            4.8274218 ,  4.7346551 ,  5.1343761 ,  5.46017944,  6.02199777,
            6.54132124,  7.27025256,  7.99780777,  8.55188217,  9.30286584,
            9.97067561, 10.81633293, 11.60615795, 12.33813552, 13.10488982,
           13.86868565, 14.60577266])



.. code:: ipython3

    ds.mean(ds.r, binby=[ds.x, ds.y], shape=32, limits=[-10, 10]) # or 2d
    ds.count(ds.r, binby=[ds.x, ds.y], shape=32, limits=[-10, 10]) # or 2d counts/histogram




.. parsed-literal::

    array([[22., 33., 37., ..., 58., 38., 45.],
           [37., 36., 47., ..., 52., 36., 53.],
           [34., 42., 47., ..., 59., 44., 56.],
           ...,
           [73., 73., 84., ..., 41., 40., 37.],
           [53., 58., 63., ..., 34., 35., 28.],
           [51., 32., 46., ..., 47., 33., 36.]])



These one and two dimensional grids can be visualized using any plotting
library, such as matplotlib, but the setup can be tedious. For
convenience we can use `plot1d <https://docs.vaex.io/en/latest/api.html#vaex.dataset.Dataset.plot1d>`__,
`plot <https://docs.vaex.io/en/latest/api.html#vaex.dataset.Dataset.plot>`__, or see the `list of
plotting commands <https://docs.vaex.io/en/latest/api.html#visualization>`__



Continue
--------

`Continue the tutorial <https://docs.vaex.io/en/latest/tutorial.html>`__ or check the
`examples <https://docs.vaex.io/en/latest/examples.html>`__

If you like vaex, please let us know by giving us a star on GitHub,

Regards,

The vaex.io team

.. |Travis| image:: https://travis-ci.org/vaexio/vaex.svg?branch=master
   :target: https://travis-ci.org/vaexio/vaex.svg?branch=master
.. |Chat| image:: https://badges.gitter.im/maartenbreddels/vaex.svg
   :alt: Join the chat at https://gitter.im/maartenbreddels/vaex
   :target: https://gitter.im/maartenbreddels/vaex?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Conda| image:: https://anaconda.org/conda-forge/vaex/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/vaex   
   

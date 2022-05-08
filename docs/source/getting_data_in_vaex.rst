Getting your data in and out of vaex
====================================

Vaex most efficiently reads hdf5 files (column based), however other datasets may be in different formats. The most flexible way to get data into vaex is to try to open your file with `TOPCAT <http://www.star.bris.ac.uk/~mbt/topcat/>`_ and export it using the colfits format. Although vaex can read these column based fits files fine, because the data is stored in big endian format (instead of the now more common little endian), which can give a 30% performance penalty.

Using the command line you can convert a (col) fits file to hdf5 and visa versa
::

 $ vaex convert file gaia-dr1.fits gaia-dr1.hdf5


Batch converting
----------------

Using TOPCAT, you can convert many files efficiently to a single colfits file from the command line, the following is an example of how to convert the full TGAS dataset into one colfits file

::

    $ wget -r --no-parent http://cdn.gea.esac.esa.int/Gaia/tgas_source/fits/
    $ find cdn.gea.esac.esa.int -iname '*.fits' > tgas.txt;
    $ topcat -stilts -Djava.io.tmpdir=/tmp tcat in=@tgas.txt out=tgas.fits ofmt=colfits
    $ vaex convert file tgas.fits tgas.hdf5 # optionally convert it to hdf5


From Python
-----------

Using the following methods, you can convert Pandas dataframes, ascii (whitespace or comma seperated) files, or numpy arrays to vaex datasets.

* `vx.from_pandas <api.html#vaex.from_pandas>`_ .
* `vx.from_ascii <api.html#vaex.from_ascii>`_ .
* `vx.from_arrays <api.html#vaex.from_arrays>`_ .
* `vx.from_astropy_table <api.html#vaex.from_astropy_table>`_ .

Then using the `vx.export_hdf5 <api.html#vaex.dataset.DatasetLocal.export_hdf5>`_ method to export it to a singe hdf5 file, e.g.:

.. code-block:: python

    import vaex as vx
    import numpy as np
    x = np.arange(0, 100)
    ds = vx.from_arrays("test-dataset", x=x, y=x**2)
    ds.export_hdf5("/tmp/test.hdf5", progress=True)

Getting your data out
---------------------

In case you have a vaex dataset, and you want to access the underlying data, they are accessible as numpy arrays using the `Dataset.columns` dictionary, or by converting them to other data structures, see for instance:

* `Dataset.to_items <api.html#vaex.dataset.Dataset.to_items>`_ .
* `Dataset.to_dict <api.html#vaex.dataset.Dataset.to_items>`_ .
* `Dataset.to_pandas_df <api.html#vaex.dataset.Dataset.to_pandas_df>`_ .
* `Dataset.to_astropy_table <api.html#vaex.dataset.Dataset.to_astropy_table>`_ .

Example:

.. code-block:: python

    import vaex as vx
    import matplotlib.pyplot as plt
    ds = vx.example()
    ds.select("x > -2")
    values = ds.to_dict(selection=True)
    plt.scatter(values["x"], values["y"])




Producing a hdf5 file
---------------------
You may want to produce an hdf5 file from you favorite language, below are a few examples how to convert data into an hdf5 file that vaex can read.


Python example
^^^^^^^^^^^^^^


This example script reads in a comma seperated values file (Example file: `helmi200.csv <https://www.astro.rug.nl/~breddels/vaex/helmi2000.csv>`_.) and outputs it to a hdf5 file that can be read by veax. Since writing the rows individually is quite slow, the rows are written in batches.

Example file: `helmi200.csv <https://www.astro.rug.nl/~breddels/vaex/helmi2000.csv>`_

.. literalinclude:: example1.py


IDL example
^^^^^^^^^^^

.. literalinclude:: ascii_to_hdf5.pro
	:language: IDL


C example
^^^^^^^^^

.. literalinclude:: ascii_to_hdf5.c
	:language: c


Examples
========


Python example
--------------


This example script reads in a comma seperated values file (Example file: `helmi200.csv <https://www.astro.rug.nl/~breddels/vaex/helmi2000.csv>`_.) and outputs it to a hdf5 file that can be read by veax. Since writing the rows individually is quite slow, the rows are written in batches.

Example file: `helmi200.csv <https://www.astro.rug.nl/~breddels/vaex/helmi2000.csv>

.. literalinclude:: example1.py


IDL example
-----------

.. literalinclude:: ascii_to_hdf5.pro
	:language: IDL


C example
-----------

.. literalinclude:: ascii_to_hdf5.c
	:language: c


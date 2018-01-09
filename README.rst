|Travis| |Conda| |Chat| 

VaeX: Visualization and eXploration
===================================

Vaex is a python library for Out-of-Core DataFrames (similar to Pandas), to visualize and explore big tabular datasets. It can calculate statistics such as mean, sum, count, standard deviation etc, on an N-dimensional grid up to a billion (10\ :sup:`9`) objects/rows per second. Visualization is done using histograms, density plots and 3d volume rendering, allowing interactive exploration of big data. Vaex uses memory mapping, zero memory copy policy and lazy computations for best performance (no memory wasted).


Vaex uses several sites:

* Main page: https://vaex.io/
* Documentation: https://docs.vaex.io/
* Github: https://github.com/maartenbreddels/vaex
* PyPi: https://pypi.python.org/pypi/vaex/

Installation
============

See https://docs.vaex.io/en/latest/installing.html or:

Using pip
::
 $ pip install --user --pre vaex

Using conda
::
 conda install -c conda-forge vaex


.. |Travis| image:: https://travis-ci.org/maartenbreddels/vaex.svg?branch=master
   :target: https://travis-ci.org/maartenbreddels/vaex
.. |Chat| image:: https://badges.gitter.im/maartenbreddels/vaex.svg
   :alt: Join the chat at https://gitter.im/maartenbreddels/vaex
   :target: https://gitter.im/maartenbreddels/vaex?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge
.. |Conda| image:: https://anaconda.org/conda-forge/vaex/badges/downloads.svg
   :target: https://anaconda.org/conda-forge/vaex   
   

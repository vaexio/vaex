|Travis| |Conda| |Chat| 

VaeX: Visualization and eXploration
===================================


Vaex is a program (and Python library) to visualize and explore large tabular datasets using statistics on an N-dimensional grid.
It mainly renders histograms, density plots and volume rendering  plots for visualization in the order of 10\ :sup:`9` rows in the order of 1 second.
For exploration it support selection in 1 and 2d, but it can also analyse the columns (dimensions) to find subspaces
which are richer in information than others.

.. image:: http://vaex.readthedocs.org/en/latest/_images/overview.png

Vaex uses several sites:

* Main page: http://vaex.astro.rug.nl/
* Github for source, bugs, wiki, releases: https://github.com/maartenbreddels/vaex
* Python Package Index for installing the source in your Python tree: https://pypi.python.org/pypi/vaex/
* Documentation, similar to the homepage, but also has older versions: http://vaex.readthedocs.org/

Installation
============

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
   

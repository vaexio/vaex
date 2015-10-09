===================================
VaeX: Visualization and eXploration
===================================




What is Vaex?
=============

Veax is a graphical tool and library to visualize and explore large tabular datasets.
It mainly renders histograms, density plots and volume rendering  plots for visualization in the order of 10\ :sup:`9` rows in the order of 1 second.
For exploration it support selection in 1 and 2d, but it can also analyse the columns (dimensions) to find subspaces
which are richer in information than others.

.. image:: images/ss1-small.png

Main features
=============

* The vaex graphical interface

    * Visualize a billion (10\ :sup:`9`) rows interactively in a graphics interface in 1d (histogram), 2d (density plot) and 3d (volume rendering)
    * Overplot vectors, for instance mean motions, tensors (for instance mean velocity dispersion tensor)
    * Custom expressions, e.g. log(sqrt(x**2+y**2)), calculated on the fly
    * publish quality output (using matplotlib)
    * Linked views:  selecting in 1 view will also select it in different views
    * data formats

     * hdf5: gadget, our own format (in the future: other formats can be supported with a few lines of code)
     * hdf5 from `Amuse <http://amusecode.org/>`_.
     * fits bintable
     * VOtable over SAMP
     * gadget native format

    * Ranking of subspaces: for 2 and 3 dimensional subspaces, a ranking can be calculated that indicates the relative richness of structure and/or correlation in them.
    * Easily showing a fraction of the data: if the rows are uncorrelated in order (random order), a subset of the data can be shown using a slider (which can make the program more responsive)
    * exporting data: the selected data can be exported for further analysis
    * undo/redo: a mistake in selection or navigation can quickly be undone using undo

* The vaex library
 * Generate the same plots and more as the graphical interface
 * Integration with IPython notebook

Demo movies
===========

.. raw:: html

    <iframe width="560" height="315" src="http://www.youtube.com/embed/oE5jN4zuhH0" frameborder="0" allowfullscreen></iframe>
    <iframe width="560" height="315" src="http://www.youtube.com/embed/An33dYPmgKI" frameborder="0" allowfullscreen></iframe>
    <iframe width="560" height="315" src="http://www.youtube.com/embed/4HHa52Gxn9w" frameborder="0" allowfullscreen></iframe>

See the :ref:`gallery` for mor examples.


Getting started
===============


If you want to try out vaex as a graphical tool only, :ref:`download the binary <installing_from_binary>` and read the :ref:`quickstart <quickstart>`.

If you want to use vaex as a library, from your script or IPython notebook, :ref:`install vaex as a library <installing_from_source`, and go through the :ref:`tutorial`.


Links
=====

Vaex uses several sites:

* Main page: http://www.astro.rug.nl/~breddels/vaex/
* Github for source, bugs, wiki, releases: https://github.com/maartenbreddels/vaex
* Python Package Index for installing the source in your Python tree: https://pypi.python.org/pypi/vaex/
* Documentation, similar to the homepage, but also has older versions: http://vaex.readthedocs.org/

Guide
=====

Contents:

.. toctree::
   :maxdepth: 2
    
   installing
   documentation
   gallery
   tipsandfaq
   examples
   credits
	




.. _installing:

Download and installing
=======================

VaeX is available as binary for OSX and Linux, without any dependencies. See the next section how to get it. For development, other plaforms, or for more optimized compiling of the source, you may want to build from source.

 * **Standalone version**: download ( `osx <https://www.astro.rug.nl/~breddels/vaex/vaex-1.0.0-beta.1-osx.zip>`_ | `linux <https://www.astro.rug.nl/~breddels/vaex/vaex-1.0.0-beta.1-linux.tar.gz>`_ )
 * **Python package**: ``pip install --pre vaex`` (no root? use ``pip install --pre --user vaex``)
 * **Anaconda users**: ``conda install -c conda-forge vaex``


.. _installing_from_binary:

Download the binary
-------------------

The binary should be fully self contained, it should not need any other software. 

OSX:
 
* `Vaex 1.0.0-beta.2-osx <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.2-osx.zip>`_ .
* `Vaex 1.0.0-beta.1-osx <https://www.astro.rug.nl/~breddels/vaex/vaex-1.0.0-beta.1-osx.zip>`_ .
* `Vaex 0.3.10-osx <https://www.astro.rug.nl/~breddels/vaex/vaex-0.3.10-osx.zip>`_ (`mirror <https://github.com/maartenbreddels/vaex/releases/download/0.3.10/vaex-0.3.10-osx.zip>`_).
* `Vaex 0.2.1-osx <https://www.astro.rug.nl/~breddels/vaex/vaex-0.2.1-osx.zip>`_.
* `Vaex 0.1.8-5-osx <https://www.astro.rug.nl/~breddels/vaex/vaex-0.1.8-5-osx.zip>`_.
* `Vaex 0.1.7-osx <https://www.astro.rug.nl/~breddels/vaex/vaex-0.1.7-osx.zip>`_.
* `Vaex 0.1.6-osx <https://www.astro.rug.nl/~breddels/vaex/vaex-osx-0.1.6.zip>`_.
* `Vaex 0.1.5-osx <https://astrodrive.astro.rug.nl/public.php?service=files&t=a408a79bc2811920878fda861f615f2a>`_.

Linux:
	
Simpy unpack the tarball and run it like:

.. code-block:: python
	
	$ tar vaex-0.3.10-linux.tar.gz
	$ ./vaex-0.3.10/vaex

* `Vaex 1.0.0-beta.2-linux <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.2-linux.tar.gz>`_.
* `Vaex 1.0.0-beta.1-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-1.0.0-beta.1-linux.tar.gz>`_.
* `Vaex 0.3.10-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.3.10-linux.tar.gz>`_ (`mirror <https://github.com/maartenbreddels/vaex/releases/download/0.3.10/vaex-0.3.10-linux.tar.gz>`_).
* `Vaex 0.2.1-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.2.1-linux.tar.gz>`_.
* `Vaex 0.1.8-5-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.1.8-5-linux.tar.gz>`_.
* `Vaex 0.1.7-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.1.7-linux.tar.gz>`_.
* `Vaex 0.1.5-linux <https://astrodrive.astro.rug.nl/public.php?service=files&t=86be18567ca6327a903f7863787c4ebf>`_

.. _installing_from_source:

From source
-----------

If for some reason the binaries above don't work, or you want to work on the source code, this method of installing is preferred. Make sure you have a Python version (2.7) with PyQt4 or PySide installed (check by running 'python -c "import PyQt4"', or 'python -c "import PySide"). If you have any issues getting PyQt working, get Anaconda (for Python 2.7) from http://continuum.io/downloads (it's a Python distribution with many packages). Also numpy and scipy may have issues installing from pip, check if you have these packages, or again go for Anaconda.

Vaex has several dependencies, besides PyQt4/PySide and possibly numpy and scipy, for all steps below the required software can be installed by running:

.. code-block:: python
	
	$ pip install -r requirements.txt

For this you need to have pip installed (try running 'pip'), or get pip from https://pip.pypa.io/en/latest/installing.html (pip makes installing easier).


Using PIP
^^^^^^^^^

* install using

 * ``pip install veax`` (if you have admin rights)
 * ``pip install vaex --user`` (this will install in the ~/.local/ directory, start by running ~/.local/bin/vaex or adding this directory to your PATH env var

From a tarball
^^^^^^^^^^^^^^

If you download the tarball from github or pypi, unpack and install by running:

* ``tar zxfv vaex-X.Y.Z.tar.gz`` (where X.Y.Z refers to the version number)
* ``cd vaex-X.Y.Z``
* install using:

 * ``python setup.py install`` (if you have admin rights)
 * ``python setup.py install --user``  (this will install in the ~/.local/ directory, start by running ~/.local/bin/vaex or adding this directory to your PATH env var

From github
^^^^^^^^^^^
* ``git clone https://github.com/maartenbreddels/vaex``
* ``cd vaex``
* install using:

 * ``python setup.py install``  (if you have admin rights)
 * ``python setup.py install --user``  (this will install in the ~/.local/ directory, start by running ~/.local/bin/vaex or adding this directory to your PATH env var

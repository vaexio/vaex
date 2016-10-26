.. _installing:

.. note::

    For the impatient:

    - If you want a standalone Python environment with vaex installed that does not interfere with you system Python, execute ``curl http://vaex.astro.rug.nl/install_conda.sh | sh -`` on your terminal.
    - To remove, execute ``rm -rf ~/miniconda-vaex ~/.condarc ~/.conda ~/.continuum``

Download and installing
=======================

The vaex program (with this you cannot do any Python programming) is available for OSX and Linux.
See the next section how to get it. For using vaex as a library, install vaex using pip or conda.

 * **Standalone version**: download ( `osx <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-osx.zip>`_ | `linux <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-linux.tar.gz>`_ )
 * **Python package**: ``pip install --pre vaex`` (no root? use ``pip install --pre --user vaex``)
 * **Anaconda users**: ``conda install -c conda-forge vaex``


.. _installing_from_binary:

Vaex program (all versions)
---------------------------

The binary should be fully self contained, it should not need any other software. 

OSX:
 
* `Vaex 1.0.0-beta.4-osx <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-osx.zip>`_ .
* `Vaex 1.0.0-beta.3-osx <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.3-osx.zip>`_ .
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

::
	
	$ tar vaex-0.3.10-linux.tar.gz
	$ ./vaex-0.3.10/vaex-0.3.10

Linux distributions with glibc 2.14 are supported, for instance Scientific Linux 6 will not work, but 7 does.

* `Vaex 1.0.0-beta.4-linux <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.4-linux.tar.gz>`_.
* `Vaex 1.0.0-beta.3-linux <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.3-linux.tar.gz>`_.
* `Vaex 1.0.0-beta.2-linux <//vaex.astro.rug.nl/program/vaex-1.0.0-beta.2-linux.tar.gz>`_.
* `Vaex 1.0.0-beta.1-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-1.0.0-beta.1-linux.tar.gz>`_.
* `Vaex 0.3.10-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.3.10-linux.tar.gz>`_ (`mirror <https://github.com/maartenbreddels/vaex/releases/download/0.3.10/vaex-0.3.10-linux.tar.gz>`_).
* `Vaex 0.2.1-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.2.1-linux.tar.gz>`_.
* `Vaex 0.1.8-5-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.1.8-5-linux.tar.gz>`_.
* `Vaex 0.1.7-linux <https://www.astro.rug.nl/~breddels/vaex/vaex-0.1.7-linux.tar.gz>`_.
* `Vaex 0.1.5-linux <https://astrodrive.astro.rug.nl/public.php?service=files&t=86be18567ca6327a903f7863787c4ebf>`_

.. _installing_from_source:


Using Anaconda
^^^^^^^^^^^^^^

It is recommended to not use your Operating system's Python (it may break your system), but install `Anaconda <http://continuum.io/downloads>`_. Anaconda is a Python distribution that makes installing Python packages much easier.

After installing Anaconda, you can install vaex using ``conda install -c maartenbreddels vaex``


Using PIP
^^^^^^^^^

If you plan to not use Anaconda, you can use pip to install vaex. However, it may be difficult to get PyQt installed. If running ``pip install PyQt5`` fails, you may want to try your favorite package manager (brew, macports) to install it instead. If you manage to run one of the following commands

* ``python -c "import PyQt4"``
* ``python -c "import PyQt5"``
* ``python -c "import PySide"``

You will probably have no trouble installing vaex, and can continue, otherwise we recommend following the Anaconda route described above.

* Continue installing vaex using

 * ``pip install --pre vaex --user`` (this will install in the ~/.local/ directory, start by running ~/.local/bin/vaex or adding this directory to your PATH env var)
 * ``pip install --pre vaex`` (if you have want to install it system wide)

Your platform may not support

From github
^^^^^^^^^^^
* ``git clone https://github.com/maartenbreddels/vaex``
* ``cd vaex``
* install using:

 * ``pip install -e --user .``  (this will install in the ~/.local/ directory, start by running ~/.local/bin/vaex or adding this directory to your PATH env var)
 * ``pip install -e .``  (if you have want to install it system wide)


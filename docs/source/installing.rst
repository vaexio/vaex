.. _installing:

Installing
==========

.. .. note::

..     For the impatient:

..     - If you want a standalone Python environment with vaex installed that does not interfere with you system Python, execute ``curl http://vaex.astro.rug.nl/install_conda.sh | bash -`` on your terminal.
..     - To remove, execute ``rm -rf ~/miniconda-vaex ~/.condarc ~/.conda ~/.continuum``


.. warning::

    It is recommended not to install directly into your operating system's Python using sudo since it may break your system. Instead, you should install `Anaconda <https://www.anaconda.com/download/>`_, which is a Python distribution that makes installing Python packages much easier or use `virtualenv or venv <https://stackoverflow.com/questions/41972261/what-is-a-virtualenv-and-why-should-i-use-one>`_.


Short version
^^^^^^^^^^^^^

 * **Anaconda users**: ``conda install -c conda-forge vaex``
 * **Regular Python users using virtualenv**: ``pip install vaex``
 * **Regular Python users (not recommended)**:  ``pip install --user vaex`` 
 * **System install (not recommended)**: ``sudo pip install vaex`` 


Longer version
^^^^^^^^^^^^^^

If you don't want all packages installed, do not install the vaex package. The vaex package is a meta packages that depends on all other vaex packages so it will instal them all, but if you don't need astronomy related parts (``vaex-astro``), or don't care about graphql (``vaex-graphql``), you can leave out those packages. Copy paste the following lines and remove what you do not need:

 * **Regular Python users**: ``pip install vaex-core vaex-viz vaex-jupyter vaex-arrow vaex-server vaex-ui vaex-hdf5 vaex-astro``
 * **Anaconda users**: ``conda install -c conda-forge vaex-core vaex-viz vaex-jupyter vaex-arrow vaex-server vaex-ui vaex-hdf5 vaex-astro``

When installing ``vaex-ui`` it does not install PyQt4, PyQt5 or PySide, you have to choose yourself and installing may be tricky. If running pip install PyQt5 fails, you may want to try your favourite package manager (brew, macports) to install it instead. You can check if you have one of these packages by running:

 * ``python -c "import PyQt4"``
 * ``python -c "import PyQt5"``
 * ``python -c "import PySide"``

For developers
^^^^^^^^^^^^^^

If you want to work on vaex for a Pull Request from the source, use the following recipe:

* ``git clone --recursive https://github.com/vaexio/vaex``  # make sure you get the submodules
* ``cd vaex``
* make sure the dev versions of pcre are installed (e.g. ``conda install -c conda-forge pcre``)
* install using:

 * ``pip install -e .``  (again, use (ana)conda or virtualenv/venv)

* If you want to do a PR

 * ``git remote rename origin upstream``
 * (now fork on github)
 * ``git remote add origin https://github.com/yourusername/vaex/``
 * ... edit code ... (or do this after the next step)
 * ``git checkout -b feature_X``
 * ``git commit -a -m "new: some feature X"``
 * ``git push origin feature_X``
 * ``git checkout master``

* Get your code in sync with upstream

 * ``git checkout master``
 * ``git fetch upstream``
 * ``git merge upstream/master``



#!/bin/bash
set -x -e

if [ -f ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc
else
    source ${HOME}/.bash_profile
fi

PYTHON_VERSION=${1:-3.7}
CONDA=${2:-conda}
conda config --set always_yes yes --set changeps1 no
conda config --remove channels defaults
conda config --add channels msys2
$CONDA update -y -q -c conda-forge $CONDA
$CONDA create -y -q -c conda-forge -n vaex-dev python=$PYTHON_VERSION
conda activate vaex-dev
# $CONDA install -y -q --file ci/conda-env-nightlies.yml -c arrow-nightlies -c conda-forge
$CONDA env update --file ci/conda-env.yml
$CONDA install -y -q compilers --file ci/conda-env-notebooks.yml -c conda-forge -c numba/label/dev

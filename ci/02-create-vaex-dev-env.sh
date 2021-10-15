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
$CONDA update -y -q -c conda-forge $CONDA
$CONDA create -y -q -c conda-forge -n vaex-dev python=$PYTHON_VERSION compilers
$CONDA activate vaex-dev
$CONDA env update --file ci/conda-env.yml
conda list

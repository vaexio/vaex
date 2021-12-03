#!/bin/bash
set -x -e

if [ -f ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc
else
    source ${HOME}/.bash_profile
fi

PYTHON_VERSION=${1:-3.7}
CONDA=${2:-conda}
conda config --remove channels defaults
conda config --set always_yes yes --set changeps1 no
$CONDA update -y -q -c conda-forge $CONDA --quiet
$CONDA create -y -q -c conda-forge -n vaex-dev python=$PYTHON_VERSION compilers --quiet
conda activate vaex-dev
$CONDA env update --file ci/conda-env.yml --quiet
conda list

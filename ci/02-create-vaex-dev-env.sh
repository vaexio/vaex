#!/bin/bash
set -x -e
PYTHON_VERSION=${1:-3.7}
CONDA=${2:-conda}
conda config --set always_yes yes --set changeps1 no
$CONDA update -y -q -c conda-forge $CONDA
$CONDA create -y -q -c conda-forge -n vaex-dev python=$PYTHON_VERSION
source activate vaex-dev
$CONDA install -y -q --file ci/conda-env-nightlies.yml -c arrow-nightlies -c conda-forge
$CONDA install -y -q compilers --file ci/conda-env.yml --file ci/conda-env-notebooks.yml -c conda-forge
$CONDA init bash

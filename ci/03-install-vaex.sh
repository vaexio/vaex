#!/bin/bash
set -e

if [ -f ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc
else
    source ${HOME}/.bash_profile
fi

conda config --set always_yes yes --set changeps1 no
conda activate vaex-dev
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CFLAGS='-Wl,-strip-all'
    export CXXFLAGS='-Wl,-strip-all'
fi
pip install myst_parser
pip install -e .

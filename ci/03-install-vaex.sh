#!/bin/bash
set -e
conda config --set always_yes yes --set changeps1 no
source activate vaex-dev
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CFLAGS='-Wl,-strip-all'
    export CXXFLAGS='-Wl,-strip-all'
fi
pip install -e .

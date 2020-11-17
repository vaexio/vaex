#!/bin/bash
set -e
conda config --set always_yes yes --set changeps1 no
source activate vaex-dev
export CFLAGS='-Wl,-strip-all'
export CXXFLAGS='-Wl,-strip-all
pip install -e .

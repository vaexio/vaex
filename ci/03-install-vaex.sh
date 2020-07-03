#!/bin/bash
set -e
conda config --set always_yes yes --set changeps1 no
source activate vaex-dev
pip install -e .

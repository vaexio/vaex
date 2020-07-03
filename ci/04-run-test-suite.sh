#!/bin/bash
set -e
PYTHON_VERSION=${1:-3.7}
source activate vaex-dev
py.test tests packages/vaex-core/vaex/test/dataset.py::TestDataset

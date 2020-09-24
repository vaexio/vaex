#!/bin/bash
set -e

source activate vaex-dev
py.test tests packages/vaex-core/vaex/test/dataset.py::TestDataset --timeout=1000

#!/bin/bash
set -e

if [ -f ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc
else
    source ${HOME}/.bash_profile
fi
export VAEX_SERVER_OVERRIDE='{"dataframe.vaex.io":"dataframe-dev.vaex.io"}'
py.test tests packages/vaex-core/vaex/datatype_test.py packages/vaex-core/vaex/file/ packages/vaex-core/vaex/test/dataset.py::TestDataset --doctest-modules packages/vaex-core/vaex/datatype.py packages/vaex-core/vaex/utils.py --timeout=1000

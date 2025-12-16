#!/bin/bash
set -e


export VAEX_SERVER_OVERRIDE='{"dataframe.vaex.io":"dataframe-dev.vaex.io"}'
python -m pytest --pyargs --doctest-modules --timeout=1000\
        tests\
        vaex.datatype_test\
        vaex.file\
        vaex.test.dataset::TestDataset\
        vaex.datatype\
        vaex.utils\
        vaex.struct\
        vaex.groupby

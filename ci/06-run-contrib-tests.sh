#!/bin/bash
set -e

if [ -f ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc
else
    source ${HOME}/.bash_profile
fi
conda activate vaex-dev

py.test packages/vaex-contrib/vaex/contrib/

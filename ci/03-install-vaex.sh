#!/bin/bash
set -e

if [ -f ${HOME}/.bashrc ]; then
    source ${HOME}/.bashrc
else
    source ${HOME}/.bash_profile
fi

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    export CFLAGS='-Wl,-strip-all'
    export CXXFLAGS='-Wl,-strip-all'
fi
pip install myst_parser
pip install -e . -v

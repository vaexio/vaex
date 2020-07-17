#!/bin/bash
set -x -e

case "$OSTYPE" in
  solaris*) echo "SOLARIS not supported" ;;
  darwin*)  wget --continue https://repo.continuum.io/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh; ;;
  linux*)   wget --continue https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh ;;
  bsd*)     echo "BSD not supported" ;;
  *)        echo "unknown: $OSTYPE not supported" ;;
esac


bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
# conda config --set always_yes yes --set changeps1 no
conda install -y -q -c conda-forge mamba -y;

#!/bin/sh
# run like $ curl -L http://bit.ly/get_vaex | sh
case "$OSTYPE" in
  solaris*) echo "SOLARIS not supported" ;;
  darwin*)  wget --continue https://repo.continuum.io/miniconda/Miniconda-latest-MacOSX-x86_64.sh -O miniconda.sh; ;;
  linux*)   wget --continue https://repo.continuum.io/miniconda/Miniconda-latest-Linux-x86_64.sh -O miniconda.sh ;;
  bsd*)     echo "BSD not supported" ;;
  *)        echo "unknown: $OSTYPE not supported" ;;
esac


bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create -q -n vaex python=2.7 numpy scipy pyqt matplotlib pyopengl h5py numexpr astropy tornado jupyter
source activate vaex
git clone https://github.com/maartenbreddels/vaex/
cd vaex
pip install -r requirements.txt
python setup.py install

echo '# to run vaex, execute: '
echo '# $ export PATH="$HOME/miniconda/bin:$PATH"; source activate vaex'
echo '# $ vaex'
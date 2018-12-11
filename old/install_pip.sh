#!/bin/sh
# run like"
# $ curl http://vaex.astro.rug.nl/install_pip.sh | bash

while [[ "$#" > 0 ]]; do case $1 in
    --dev) dev=1;;
    --python) python=$2; shift;;
    *) break;;
  esac; shift;
done


pyversion=${python:-"3"}
echo "python version set to $pyversion"
PY="python$pyversion"
PIP="pip$pyversion"


wget --continue https://bootstrap.pypa.io/get-pip.py
$PY get-pip.py --user
export PATH=$HOME/.local/bin:$PATH
$PIP install virtualenv --user
virtualenv vaex-env
source vaex-env/bin/activate
$PIP install numpy

if [ -z ${dev+x} ]; then
    echo "installing latest released version";
    $PIP install --pre vaex;
else
    echo "installing development version";
    git clone https://github.com/maartenbreddels/vaex/;
    $PIP install -r vaex/requirements.txt;
    $PIP install -e ./vaex;
fi

echo "============================================="
echo '# Attempt to install PyQt5, if this fails you may want to switch to Vanilla Python 3.5 for Linux, or use anaconda: '
echo "============================================="
$PIP install PyQt5

echo "============================================="
echo '# to run vaex, execute: '
echo "# $ source $PWD/vaex-env/bin/activate"
echo '# $ vaex'
echo '# for future use, you may want to put this in your .bashrc (.bash_profile on OSX)'
echo "============================================="

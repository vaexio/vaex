#!/bin/sh
# run like"
# $ curl http://vaex.astro.rug.nl/install_pip.sh | bash

while [[ "$#" > 0 ]]; do case $1 in
    --dev) dev=1;;
    --python) python=$2; shift;;
    *) break;;
  esac; shift;
done

wget --continue https://bootstrap.pypa.io/get-pip.py
python get-pip.py --user
export PATH=$HOME/.local/bin:$PATH
pip install virtualenv --user
virtualenv vaex-env
source vaex-env/bin/activate
pip install numpy

if [ -z ${dev+x} ]; then
    echo "installing latest released version"
    pip install --pre vaex
else
    echo "installing development version"
    git clone https://github.com/maartenbreddels/vaex/
    pip install -r vaex/requirements.txt
    pip install -e ./vaex
fi


echo "============================================="
echo '# to run vaex, execute: '
echo "# $ source $PWD/vaex-env/bin/activate'
echo '# $ vaex'
echo '# for future use, you may want to put this in your .bashrc (.bash_profile on OSX)'
echo "============================================="

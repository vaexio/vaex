#!/bin/bash
set -x -e
apt-get update
apt-get install -y -q wget
./ci/01-setup-conda.sh
export PATH="$HOME/miniconda/bin:$PATH"
./ci/02-create-vaex-dev-env.sh 3.7 mamba
./ci/03-install-vaex.sh
./ci/04-run-test-suite.sh
./ci/05-run-notebooks.sh

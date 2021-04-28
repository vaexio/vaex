#!/bin/bash
set -e
source activate vaex-dev
python -m pip install healpy
cd docs/source
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute tutorial_jupyter.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute tutorial.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute tutorial_ml.ipynb --ExecutePreprocessor.timeout=240
# we cannot run the arrow example, maybe with a remote dataframe?
# python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_arrow.ipynb
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_dask.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_graphql.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_io.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_dask.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_ml_iris.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_jupyter_ipyvolume.ipynb --ExecutePreprocessor.timeout=240
python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --to html --execute example_jupyter_plotly.ipynb --ExecutePreprocessor.timeout=240
# this fails to run currently
# python -m nbconvert --TagRemovePreprocessor.remove_cell_tags="('skip-ci',)" --execute example_ml_titanic.ipynb

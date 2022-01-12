import vaex
import vaex.ml
import vaex.datasets
import vaex.ml.ui


def test_widgetize():
    # Load datasets
    ds = vaex.datasets.iris()
    # define the list of transformers
    transformer_list = [vaex.ml.StandardScaler(),
                        vaex.ml.MinMaxScaler(),
                        # vaex.ml.PCA(), PCA is disabled for now, because n_componts can be None
                        vaex.ml.LabelEncoder(),
                        vaex.ml.OneHotEncoder(),
                        vaex.ml.MaxAbsScaler(),
                        vaex.ml.RobustScaler()]
    # Go through all the transformers
    for i, v in enumerate(transformer_list):
        vaex.ml.ui.Widgetize(v, ds)

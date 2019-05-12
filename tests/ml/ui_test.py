import vaex
import vaex.ml
import vaex.ml.datasets
import vaex.ml.ui


def test_widgetize():
    # Load datasets
    ds = vaex.ml.datasets.load_iris()
    # define the list of transformers
    transformer_list = [vaex.ml.StandardScaler(),
                        vaex.ml.MinMaxScaler(),
                        vaex.ml.PCA(),
                        vaex.ml.LabelEncoder(),
                        vaex.ml.OneHotEncoder(),
                        vaex.ml.MaxAbsScaler(),
                        vaex.ml.RobustScaler()]
    # Go through all the transformers
    for i, v in enumerate(transformer_list):
        vaex.ml.ui.Widgetize(v, ds)

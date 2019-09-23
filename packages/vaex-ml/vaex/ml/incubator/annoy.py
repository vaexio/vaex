import base64
import tempfile

import traitlets
import numpy as np
import annoy

import vaex
import vaex.serialize
from vaex.ml import state
from .. import generate


@vaex.serialize.register
@generate.register
class ANNOYModel(state.HasState):
    features = traitlets.List(traitlets.Unicode(), help='List of features to use.')
    n_trees = traitlets.Int(default_value=10, help="Number of trees to build.")
    search_k = traitlets.Int(default_value=-1, help='Jovan?')
    n_neighbours = traitlets.Int(default_value=10, help='Now many neighbours')
    metric = traitlets.Enum(values=['euclidean', 'manhattan', 'angular', 'hamming', 'dot'], default_value='euclidean', help='Metric to use for distance calculations')
    predcition_name = prediction_name = traitlets.Unicode(default_value='annoy_prediction', help='Output column name for the neighbours when transforming a DataFrame')

    def __call__(self, *args):
        data2d = np.vstack([arg.astype(np.float64) for arg in args]).T.copy()
        result = []
        for i in range(len(data2d)):
            result.append(self.index_builder.get_nns_by_vector(n=self.n_neighbours, vector=data2d[i], search_k=self.search_k))
        return np.array(result)

    def fit(self, dataframe):
        self.index_builder = annoy.AnnoyIndex(len(self.features), metric=self.metric)
        for i in range(len(dataframe)):
            self.index_builder.add_item(i, dataframe[self.features][i])
        self.index_builder.build(self.n_trees)

    def transform(self, dataframe):
        copy = dataframe.copy()
        lazy_function = copy.add_function('annoy_get_nns_by_vector_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_name, expression, unique=False)
        return copy

    def predict(self, dataframe, n_neighbours=None, search_k=None):
        search_k = search_k or self.search_k
        n_neighbours = n_neighbours or self.n_neighbours
        result = np.zeros((len(dataframe), n_neighbours), dtype=np.int)
        for i in range(len(dataframe)):
            result[i] = self.index_builder.get_nns_by_vector(n=n_neighbours, search_k=search_k, vector=dataframe[self.features][i])
        return result

    def state_get(self):
        filename = tempfile.mktemp()
        self.index_builder.save(filename)
        with open(filename, 'rb') as f:
            data = f.read()
        return dict(tree_state=base64.encodebytes(data).decode('ascii'),
                    substate=super(ANNOYModel, self).state_get(),
                    n_dimensions=len(self.features))

    def state_set(self, state, trusted=True):
        super(ANNOYModel, self).state_set(state['substate'])
        data = base64.decodebytes(state['tree_state'].encode('ascii'))
        n_dimensions = state['n_dimensions']
        filename = tempfile.mktemp()
        with open(filename, 'wb') as f:
            f.write(data)
        self.index_builder = annoy.AnnoyIndex(n_dimensions)
        self.index_builder.load(filename)
        return self.index_builder


if __name__ == "__main__":
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split()
    features = ds_train.column_names[:4]
    m = ANNOYModel(features=features, n_trees=50)
    m.fit(ds_train)
    m.predict(ds_test)
    m.transform(ds_test)

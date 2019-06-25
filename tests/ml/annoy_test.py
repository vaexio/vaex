import pytest
pytest.importorskip("annoy")

import annoy
import vaex.ml.incubator.annoy
import vaex.ml.datasets
import test_utils


@test_utils.skip_incubator
def test_annoy():
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(0.05)
    features = ds_train.column_names[:4]
    model = vaex.ml.incubator.annoy.ANNOYModel(features=features, n_neighbours=5, metric='euclidean', n_trees=100)
    model.fit(ds_train)
    ds_train = model.transform(ds_test)
    ds_test = model.transform(ds_test)
    pred = model.predict(ds_test)

    assert pred.shape[1] == 5
    assert pred.tolist() == ds_test.annoy_prediction.values.tolist()


@test_utils.skip_incubator
def test_annoy_validation():
    '''
    Annoy is known to experience problems with seeding and random states.
    Thus to compare results, the annoy algorithm needs to be ran in separate contexts.
    Please see https://github.com/spotify/annoy/issues/188 for more details.
    '''
    def vaex_annoy():  # annoy from within vaex
        ds = vaex.ml.datasets.load_iris()
        ds_train, ds_test = ds.ml.train_test_split(0.05, verbose=False)
        features = ds_train.column_names[:4]
        model = vaex.ml.incubator.annoy.ANNOYModel(features=features, n_neighbours=5, metric='euclidean', n_trees=100)
        model.fit(ds_train)
        pred = model.predict(ds_test)
        return pred.tolist()

    def annoy_annoy():  # the stand-alone annoy
        ds = vaex.ml.datasets.load_iris()
        ds_train, ds_test = ds.ml.train_test_split(0.05, verbose=False)
        features = ds_train.column_names[:4]
        index = annoy.AnnoyIndex(4, metric='euclidean')
        for i in range(len(ds_train)):
            index.add_item(i, ds_train[features][i])
        index.build(100)

        annoy_results = []
        for i in range(len(ds_test)):
            annoy_results.append(index.get_nns_by_vector(n=5, vector=ds_test[features][i]))

        return annoy_results

    assert vaex_annoy() == annoy_annoy()


@test_utils.skip_incubator
def test_annoy_serialize():
    ds = vaex.ml.datasets.load_iris()
    ds_train, ds_test = ds.ml.train_test_split(0.05, verbose=False)
    features = ds_train.column_names[:4]
    model = vaex.ml.incubator.annoy.ANNOYModel(features=features, n_neighbours=5, metric='euclidean', n_trees=100)

    model.fit(ds_train)

    # simply test if state_get/set works
    state = model.state_get()
    model.state_set(state)

    # test the full state transfer
    ds_train = model.transform(ds_test)
    state = ds_train.state_get()
    ds_test.state_set(state)

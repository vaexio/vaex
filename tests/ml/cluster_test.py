import pytest
import tempfile
import numpy as np
import vaex.ml.cluster
import vaex.ml.datasets


init = np.array([[0, 1/5], [1.2/2, 4/5], [2.5/2, 6/5]])
features = ['petal_width/2', 'petal_length/5']


def test_serialize():
    df = vaex.ml.datasets.load_iris()
    kmeans = vaex.ml.cluster.KMeans(n_clusters=3, features=features, init='random', random_state=42, max_iter=1)
    kmeans.fit(df)

    df_k = kmeans.transform(df)
    class_1 = df_k.evaluate(kmeans.prediction_label)

    path = tempfile.mktemp('.yaml')
    pipeline = vaex.ml.Pipeline([kmeans])
    pipeline.save(path)

    pipeline = vaex.ml.Pipeline()
    pipeline.load(path)
    pipeline.transform(df)

    class_2 = df_k.evaluate(kmeans.prediction_label)
    np.testing.assert_allclose(class_1, class_2)


def test_kmeans_random_state():
    df = vaex.ml.datasets.load_iris()
    # TODO: make init take a ndarray
    kmeans = vaex.ml.cluster.KMeans(n_clusters=3, features=features, init='random', random_state=42, max_iter=1)
    kmeans.fit(df)
    inertia = kmeans.inertia
    cluster_centers = kmeans.cluster_centers

    # check we get the same result
    kmeans = vaex.ml.cluster.KMeans(n_clusters=3, features=features, init='random', random_state=42, max_iter=1)
    kmeans.fit(df)
    assert kmeans.inertia == inertia
    assert kmeans.cluster_centers == cluster_centers

    # more iterations should give better intertia
    kmeans = vaex.ml.cluster.KMeans(n_clusters=3, features=features, init='random', random_state=42, max_iter=2)
    kmeans.fit(df)
    assert kmeans.inertia < inertia
    # assert kmeans.cluster_centers != cluster_centers


@pytest.mark.parametrize("max_iter", [1, 2, 10])
def test_kmeans(max_iter):
    df = vaex.ml.datasets.load_iris()
    # TODO: make init take a ndarray
    kmeans_vaex = kmeans = vaex.ml.cluster.KMeans(n_clusters=3, features=features,
                                                  init=init.tolist(), max_iter=max_iter, verbose=True, n_init=3)
    kmeans.fit(df)
    centers = kmeans.cluster_centers

    df_k = kmeans.transform(df)

    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=3, init=init, max_iter=max_iter, n_init=1)
    X = np.array(df[features])
    kmeans.fit(X)
    centers_sk = kmeans.cluster_centers_.tolist()
    np.testing.assert_allclose(centers_sk, centers, atol=0.1 if max_iter > 1 else 1e-5, rtol=0)
    if max_iter == 1:  # for max_iter 1 we should have the same answer
        class_vaex = df_k.evaluate(kmeans_vaex.prediction_label)
        class_sk = kmeans.predict(X)
        print(">", class_vaex, kmeans_vaex.prediction_label)
        print(">", class_sk)
        np.testing.assert_allclose(class_sk, class_vaex)

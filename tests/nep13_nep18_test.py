import vaex
import pytest
import numpy as np
import contextlib
import sklearn
import sklearn.preprocessing.data
import sklearn.preprocessing._encoders
import sklearn.preprocessing.data
import sklearn.preprocessing._encoders
import sklearn.decomposition.pca
# import sklearn.preprocessing.base
import numpy.linalg
import dask
import dask.array as da
from vaex.ml import sklearn_patch


@pytest.fixture
def df():
    x = np.arange(1, 11, dtype=np.float64)
    y = x**2
    df = vaex.from_arrays(x=x, y=y)
    #df['z'] = df.x + df.y
    return df


@pytest.mark.parametrize("transpose", [True, False])
def test_basic(df, transpose):
    A = np.array(df[['x', 'y']])
    view = df.numpy[:, [0, 1]]
    if transpose:
        A = A.T
        view = view.T
    assert A.shape == view.shape == view.numpy.shape
    assert A.dtype == view.dtype
    assert A.ndim == view.ndim == 2
    assert len(A) == len(A)


@pytest.mark.parametrize("transpose", [True, False])
def test_assignment_views(df, transpose):
    y = df.y.values
    view = df.numpy[:, 0]
    view[:] = 1
    assert np.all(df.x.values == 1)
    assert np.all(df.y.values == y)
    view = df.numpy[:, [0]]
    view[:] = 2
    assert np.all(df.x.values == 2)
    assert np.all(df.y.values == y)
    view = df.numpy[:, [0, 1]]
    view[:, :] = 3
    assert np.all(df.x.values == 3)
    assert np.all(df.y.values == 3)
    view = df.numpy[:, [0, 1]][:,[1]]
    view[:, :] = 4
    assert np.all(df.x.values == 3)
    assert np.all(df.y.values == 4)
    view = df.numpy[:, [0, 1]].T
    view[0] = 5
    assert np.all(df.x.values == 5)
    assert np.all(df.y.values == 4)

    view[[1, 0]] = 6
    assert np.all(df.x.values == 6)
    assert np.all(df.y.values == 6)



def test_assign_df(df):
    A = np.array(df)
    df2 = df + 1
    A2 = np.array(df2)
    df3 = df.copy()
    df3.numpy[:,:] = df2
    assert np.array(df3).tolist() == A2.tolist()
    assert np.array(df3.T).tolist() == A2.T.tolist()


def test_binary_scalar(df):
    x = df.x.values
    y = df.y.values
    x2 = x + 3
    y2 = y + 3
    assert (df + 3).x.tolist() == x2.tolist()
    assert (df + 3).y.tolist() == y2.tolist()
    assert (df.T + 3).df.x.tolist() == x2.tolist()
    assert (df.T + 3).df.y.tolist() == y2.tolist()


def test_binary_array(df):
    A = np.array(df)
    x = df.x.values
    A2 = A.T + x
    df2 = (df.T + x).df
    assert np.array(df2).tolist() == A2.T.tolist()
    assert np.array(df2.T).tolist() == A2.tolist()


def test_binary_df(df):
    A = np.array(df)
    df2 = df + 1
    A2 = np.array(df2)
    df3 = df + df2
    A3 = A + A2
    assert np.array(df3).tolist() == A3.tolist()
    assert np.array(df3.T).tolist() == A3.T.tolist()


def test_binary_df_broadcast(df):
    A = np.array(df)
    X = A[:, 0:1] + 1
    df2 = df + 1
    df3 = df.numpy + df2.numpy[:, 0:1]
    A3 = A + X
    assert np.array(df3).tolist() == A3.tolist()
    assert np.array(df3.T).tolist() == A3.T.tolist()

    df3 = df2.numpy[:, 0:1] + df.numpy
    A3 = X + A
    assert np.array(df3).tolist() == A3.tolist()
    assert np.array(df3.T).tolist() == A3.T.tolist()


def test_binary_df_out(df):
    A = np.array(df)
    df2 = df + 1
    A2 = np.array(df2)
    df3 = df * 0
    np.add(df, df2, out=df3.numpy[:, [0, 1]])
    A3 = A + A2
    assert np.array(df3).tolist() == A3.tolist()
    assert np.array(df3.T).tolist() == A3.T.tolist()


def test_clip(df):
    X = np.array(df)
    Xc = np.clip(X, 0, 4)
    c = np.clip(df.numpy, 0, 4)
    assert Xc.tolist() == np.array(c).tolist()

    Xc = np.clip(X, 0, [4, 10])
    c = np.clip(df.numpy, 0, [4, 10], out=df)
    assert Xc.tolist() == np.array(c).tolist()


def test_aggregates(df):
    A = np.array(df)
    a = np.nanmax(A)
    passes = df.executor.passes
    assert a.tolist() == np.nanmax(df).tolist()
    assert df.executor.passes == passes + 1, "aggregation should be done in 1 pass"

    a = np.nanmax(A, axis=0)
    assert a.tolist() == np.nanmax(df, axis=0).tolist()
    a = np.nanmax(A.T, axis=1)
    assert a.tolist() == np.nanmax(df.T, axis=1).tolist()


@pytest.mark.xfail
def test_aggregates_columnwise(df):
    # similar to test_aggregates, but now it should go over the columns
    A = np.array(df)
    assert isinstance(np.nanmax(df, axis=1), vaex.Expression)
    assert a.tolist() == np.nanmax(df, axis=1).tolist()
    a = np.nanmax(A.T, axis=0)
    assert a.tolist() == np.nanmax(df.T, axis=0).tolist()


def test_zeros_like(df):
    z = np.zeros_like(df.x)
    assert z.tolist() == [0] * 10


def test_mean(df):
    means = np.mean(df)
    assert means[0] == df.x.mean()


def test_ufuncs(df):
    assert np.log(df).x.tolist() == df.x.log().tolist()
    assert np.log(df.T).df.x.tolist() == df.x.log().tolist()


def test_unary(df):
    assert (-df).x.tolist() == (-df.x.values).tolist()
    assert (np.negative(df)).x.tolist() == (-df.x.values).tolist()
    assert (-df.T).df.x.tolist() == (-df.x.values).tolist()
    assert (np.negative(df.T)).df.x.tolist() == (-df.x.values).tolist()


def test_dot(df):
    x = df.x.values
    y = df.y.values
    X = np.array(df)
    print(X.shape)
    Y = X.dot([[1, 0], [0, 1]])
    assert np.all(Y[:,0] == x)
    assert np.all(Y[:,1] == y)

    # not repeat with vaex
    with sklearn_patch(), df.array_casting_disabled():
        df_dot = np.dot(df, [[1, 0], [0, 1]])
    df_dot._allow_array_casting = True
    Yv = np.array(df_dot)
    assert np.all(Yv[:, 0] == x)
    assert np.all(Yv[:, 1] == y)

    # check order
    Y = X.dot([[1, 1], [-1, 1]])
    assert np.all(Y[:, 0] == x - y)
    assert np.all(Y[:, 1] == y + x)

    with sklearn_patch(), df.array_casting_disabled():
        df_dot = np.dot(df, [[1, 1], [-1, 1]])
    df_dot._allow_array_casting = True
    Yv = np.array(df_dot)
    assert np.all(Yv[:,0] == x - y)
    assert np.all(Yv[:,1] == y + x)

    # check non-square
    Y = X.dot([[1], [-1]])
    assert np.all(Y[:,0] == x - y)

    with sklearn_patch(), df.array_casting_disabled():
        df_dot = np.dot(df, [[1], [-1]])
    df_dot._allow_array_casting = True
    Yv = np.array(df_dot)
    assert np.all(Yv[:,0] == x - y)



def test_sklearn_min_max_scalar(df):
    from sklearn.preprocessing import MinMaxScaler
    with sklearn_patch(), df.array_casting_disabled():
        scaler = MinMaxScaler()
        scaler.fit(df)

        dft = scaler.transform(df)
        assert isinstance(dft, vaex.DataFrame)
    dft._allow_array_casting = True
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))


def test_sklearn_standard_scaler(df):
    from sklearn.preprocessing import StandardScaler
    with sklearn_patch(), df.array_casting_disabled():
        scaler = StandardScaler()
        scaler.fit(df)

        dft = scaler.transform(df)
        assert isinstance(dft, vaex.DataFrame)
    dft._allow_array_casting = True
    X = np.array(df)
    Xt = scaler.transform(X)
    assert np.all(Xt == np.array(dft))


@pytest.mark.parametrize("standardize", [True, False])
@pytest.mark.parametrize("method", ['yeo-johnson', 'box-cox'])
def est_sklearn_power_transformer(df, standardize, method):
    from sklearn.preprocessing import PowerTransformer
    with sklearn_patch(), df.array_casting_disabled():
        power_trans_vaex = PowerTransformer(standardize=standardize, method=method, copy=True)
        dft = power_trans_vaex.fit_transform(df)
        assert isinstance(dft, vaex.DataFrame)

    dft._allow_array_casting = True
    X = np.array(df)
    power_trans_sklearn = PowerTransformer(standardize=standardize, method=method, copy=True)
    Xt = power_trans_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)


@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_polynomial_transformer(df, degree, interaction_only, include_bias):
    with sklearn_patch(), df.array_casting_disabled():
        poly_trans_vaex = sklearn.preprocessing.PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
        dft = poly_trans_vaex.fit_transform(df.numpy)
        assert isinstance(dft, vaex.numpy.DataFrameAccessorNumpy)

    poly_trans_sklearn = sklearn.preprocessing.PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)
    X = np.array(df)
    Xt = poly_trans_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)


@pytest.mark.skip(reason='not supported yet')
@pytest.mark.parametrize("output_distribution", ['uniform', 'normal'])
def test_quantile_transformer(df, output_distribution):
    with sklearn_patch(), df.array_casting_disabled():
        quant_trans_vaex = sklearn.preprocessing.QuantileTransformer(n_quantiles=5, random_state=42, output_distribution=output_distribution)
        dft = quant_trans_vaex.fit_transform(df)
        assert isinstance(dft, vaex.DataFrame)

    quant_trans_sklearn = sklearn.preprocessing.QuantileTransformer(n_quantiles=5, random_state=42, output_distribution=output_distribution)
    X = np.array(df)
    Xt = quant_trans_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)


@pytest.mark.parametrize("n_bins", [2, 3])
# @pytest.mark.parametrize("encode", ['ordinal', 'onehot-dense'])
@pytest.mark.parametrize("encode", ['ordinal'])
# @pytest.mark.parametrize("strategy", ['uniform', 'quantile', 'kmeans'])
@pytest.mark.parametrize("strategy", ['uniform'])
def test_kbins_discretizer(df, n_bins, encode, strategy):
    with sklearn_patch(), df.array_casting_disabled():
        trans = sklearn.preprocessing.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
        dft = trans.fit_transform(df.numpy)
        # assert isinstance(dft, vaex.DataFrame)

    trans_sklearn = sklearn.preprocessing.KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
    X = np.array(df)
    Xt = trans_sklearn.fit_transform(X)
    dft._allow_array_casting = True
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=3)

@pytest.mark.parametrize("n_components", [1, 2])
# @pytest.mark.parametrize("svd_solver", ['full', 'arpack', 'randomized'])
@pytest.mark.parametrize("svd_solver", ['full'])
def test_sklearn_pca(df, n_components, svd_solver):
    from sklearn.decomposition import PCA
    with sklearn_patch(), df.array_casting_disabled():
        pca_vaex = PCA(n_components=n_components, random_state=42, svd_solver=svd_solver)
        dft = pca_vaex.fit_transform(df)

    pca_sklearn = PCA(n_components=n_components, random_state=42, svd_solver=svd_solver)
    X = np.array(df)
    Xt = pca_sklearn.fit_transform(X)
    np.testing.assert_array_almost_equal(Xt, np.array(dft), decimal=6)


def test_dask_qr(df):
    X = np.array(df)
    A, b = np.linalg.qr(X)
    Av, bv = np.linalg.qr(df)

    assert b.tolist() == bv.tolist()
    assert A.tolist() == Av.tolist()

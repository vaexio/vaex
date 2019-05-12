import os
import vaex
import numpy as np


def load_iris():
    '''Load and return the iris dataset (classification).

    The iris dataset is a classic and very easy multi-class classification dataset.

    =================   ==============
    Classes                          3
    Samples per class               50
    Samples total                  150
    Dimensionality                   4
    Features            real, positive
    =================   ==============

    Example:
    ========

    >>> import vaex.ml
    >>> df = vaex.ml.datasets.load_iris()
    >>> df.describe()
    '''
    dirname = os.path.dirname(__file__)
    return vaex.open(os.path.join(dirname, 'iris.hdf5'))


def _iris(name, iris_previous, N):
    filename = os.path.join(vaex.utils.get_private_dir('data'), name + '.hdf5')
    if os.path.exists(filename):
        return vaex.open(filename)
    else:
        iris = iris_previous()
        repeat = int(np.ceil(N / len(iris)))
        ds = vaex.dataset.DatasetConcatenated([iris] * repeat)
        ds.export_hdf5(filename)
        return vaex.open(filename)


def iris_subsample(N, error_percentage=5, ds=None):
    '''Returns the iris set repeated so it include ~1e4 rows'''
    # return _iris_subsample('iris_1e4', iris, int(1e4))
    ds = ds or load_iris()
    ds_out = None
    repeats = int(np.ceil(N / len(ds)))
    for feature in ds.get_column_names():
        if feature in ['random_index']:
            continue
        data = ds[feature].values
        min, max = ds.minmax(feature)
        error = (max - min)/100*error_percentage
        data_out = np.repeat(data, repeats)
        if feature not in ['class_']:
            data_out += np.random.random(len(data_out)) * error
        if ds_out is None:
            ds_out = vaex.from_arrays(feature=data_out)
        else:
            ds_out.add_column(feature, data_out)
    return ds_out


def load_iris_1e4():
    '''Returns the iris set repeated so it include ~1e4 rows'''
    return _iris('iris_1e4', load_iris, int(1e4))


def load_iris_1e5():
    '''Returns the iris set repeated so it include ~1e5 rows'''
    return _iris('iris_1e5', load_iris_1e4, int(1e5))


def load_iris_1e6():
    '''Returns the iris set repeated so it include ~1e6 rows'''

    return _iris('iris_1e6', load_iris_1e5, int(1e6))


def load_iris_1e7():
    '''Returns the iris set repeated so it include ~1e7 rows'''
    return _iris('iris_1e7', load_iris_1e6, int(1e7))


def load_iris_1e8():
    '''Returns the iris set repeated so it include ~1e8 rows'''
    return _iris('iris_1e8', load_iris_1e7, int(1e8))


def load_iris_1e9():
    '''Returns the iris set repeated so it include ~1e8 rows'''
    return _iris('iris_1e9', load_iris_1e8, int(1e9))


def load_titanic():
    '''
    Returns the classic Titanic dataset.

    Description of the columns can be found in dataset.description.

    Example:
    ========

    >>> import vaex.ml
    >>> df = vaex.mk.dataset.load_titanic()
    >>> print(df.description)
    >>> df.describe()
    '''
    dirname = os.path.dirname(__file__)
    return vaex.open(os.path.join(dirname, 'titanic.hdf5'))

import pytest
import numpy as np
pytest.importorskip("sklearn.metrics")
from sklearn import metrics

import vaex.datasets
import vaex.ml
import vaex.ml.metrics
from vaex.ml.metrics import ensure_string_arguments

df_binary = vaex.from_arrays(x=[1, 0, 1, 1, 0, 0, 0],
                             y=[1, 0, 1, 1, 0, 1, 1])

df_multi_class = vaex.from_arrays(x=[1, 0, 1, 1, 0, 2, 0, 2],
                                  y=[1, 0, 1, 1, 0, 2, 2, 1])

df_multi_class_strings = vaex.from_arrays(x=['dog', 'cat', 'dog', 'dog', 'cat', 'mouse', 'cat', 'mouse'],
                                          y=['dog', 'cat', 'dog', 'dog', 'cat', 'mouse', 'mouse', 'dog'])


def test_ensure_string_arguments():
    df = vaex.datasets.iris()
    assert ensure_string_arguments(df.class_) == ['class_']
    set(ensure_string_arguments(df.class_, df.sepal_length)) == set(['class_', 'sepal_length'])
    set(ensure_string_arguments('petal_width', df.sepal_length)) == set(['petal_width', 'sepal_length'])


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
def test_accuracy_score(df):
    assert df.ml.metrics.accuracy_score(df.x, df.y) == metrics.accuracy_score(df.x.values, df.y.values)


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
def test_precision_score(df):
    average = 'binary'
    average = 'binary'
    if df.x.nunique() > 2:
        average = 'macro'

    vaex_score = df.ml.metrics.precision_score(df.x, df.y, average=average)
    sklearn_score = metrics.precision_score(df.x.values, df.y.values, average=average)
    assert vaex_score == sklearn_score


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
def test_recall_score(df):
    average = 'binary'
    if df.x.nunique() > 2:
        average = 'macro'

    vaex_score = df.ml.metrics.recall_score(df.x, df.y, average=average)
    sklearn_score = metrics.recall_score(df.x.values, df.y.values, average=average)
    assert vaex_score == sklearn_score


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
def test_f1_score(df):
    average = 'binary'
    if df.x.nunique() > 2:
        average = 'macro'

    vaex_score = df.ml.metrics.f1_score(df.x, df.y, average=average)
    sklearn_score = metrics.f1_score(df.x.values, df.y.values, average=average)
    np.testing.assert_almost_equal(vaex_score, sklearn_score)


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
def test_matthews_correlation_coefficient(df):
    vaex_score = df.ml.metrics.matthews_correlation_coefficient(df.x, df.y)
    sklearn_score = metrics.matthews_corrcoef(df.x.values, df.y.values)
    assert vaex_score == sklearn_score


@pytest.mark.parametrize('df', [df_binary, df_multi_class])
def test_confusion_matrix(df):
    vaex_result = df.ml.metrics.confusion_matrix(df.x, df.y)
    sklearn_result = metrics.confusion_matrix(df.x.values, df.y.values)
    np.testing.assert_array_equal(vaex_result, sklearn_result)


def test_mean_absolute_error():
    df = vaex.datasets.iris()
    vaex_result = df.ml.metrics.mean_absolute_error(df.petal_width, df.petal_length)
    sklearn_result = metrics.mean_absolute_error(df.petal_width.values, df.petal_length.values)
    assert vaex_result == sklearn_result


def test_mean_squared_error():
    df = vaex.datasets.iris()
    vaex_result = df.ml.metrics.mean_squared_error(df.petal_width, df.petal_length)
    sklearn_result = metrics.mean_squared_error(df.petal_width.values, df.petal_length.values)
    assert vaex_result == sklearn_result


def test_r2_score():
    df = vaex.datasets.iris()
    vaex_result = df.ml.metrics.r2_score(df.petal_width, df.petal_length)
    sklearn_result = metrics.r2_score(df.petal_width.values, df.petal_length.values)
    np.testing.assert_array_almost_equal(vaex_result, sklearn_result, decimal=7)


def test_classification_report():
    report = df_binary.ml.metrics.classification_report(df_binary.x, df_binary.y)
    expected_result = '\n        Classification report:\n\n        Accuracy:  0.714\n        Precision: 0.6\n        Recall:    1.0\n        F1:        0.75\n        '
    assert report == expected_result

    report = df_multi_class.ml.metrics.classification_report(df_binary.x, df_binary.y, average='macro')
    expected_result = '\n        Classification report:\n\n        Accuracy:  0.75\n        Precision: 0.75\n        Recall:    0.722\n        F1:        0.719\n        '
    assert report == expected_result

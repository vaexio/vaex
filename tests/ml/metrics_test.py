import pytest
import numpy as np
pytest.importorskip("sklearn.metrics")
from sklearn import metrics

import vaex.datasets
import vaex.ml
import vaex.ml.metrics
from vaex.ml.metrics import ensure_string_arguments

df_binary = vaex.from_arrays(x=[1, 0, 1, 1, 0, 0, 0],
                             y=[1, 0, 1, 1, 0, 1, 1],
                             z=[0, 1, 0, 0, 0, 1, 1])

df_multi_class = vaex.from_arrays(x=[1, 0, 1, 1, 0, 2, 0, 2, 1],
                                  y=[1, 0, 1, 1, 0, 2, 2, 1, 1],
                                  z=[0, 1, 0, 0, 0, 1, 1, 0, 1])

df_multi_class_strings = vaex.from_arrays(x=['dog', 'cat', 'dog', 'dog', 'cat', 'mouse', 'cat', 'mouse'],
                                          y=['dog', 'cat', 'dog', 'dog', 'cat', 'mouse', 'mouse', 'dog'],
                                          z=[0, 0, 0, 1, 1, 0, 1, 0])


def test_ensure_string_arguments():
    df = vaex.datasets.iris()
    assert ensure_string_arguments(df.class_) == ['class_']
    set(ensure_string_arguments(df.class_, df.sepal_length)) == set(['class_', 'sepal_length'])
    set(ensure_string_arguments('petal_width', df.sepal_length)) == set(['petal_width', 'sepal_length'])


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
@pytest.mark.parametrize('selection', [None, [None, 'z==0']])
def test_accuracy_score(df, selection):
    if selection is None:
        assert df.ml.metrics.accuracy_score(df.x, df.y) == metrics.accuracy_score(df.x.values, df.y.values)
    else:
        result_vaex = df.ml.metrics.accuracy_score(df.x, df.y, selection=selection)
        for i, sel in enumerate(selection):
            result_sklearn = metrics.accuracy_score(df.filter(sel).x.values, df.filter(sel).y.values)
            assert result_vaex[i] == result_sklearn


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
@pytest.mark.parametrize('selection', [None, [None, 'z==0', 'z==1']])
def test_precision_recall_f1_score(df, selection):
    average = 'binary'
    if df.x.nunique() > 2:
        average = 'macro'

    if selection is None:
        vaex_score = df.ml.metrics.precision_score(df.x, df.y, average=average)
        sklearn_score = metrics.precision_score(df.x.values, df.y.values, average=average)
        assert vaex_score == sklearn_score

    else:
        vaex_score = df.ml.metrics.precision_recall_fscore(df.x, df.y, selection=selection, average=average)
        vaex_score = np.array(vaex_score).T
        for i, sel in enumerate(selection):
            sklearn_score = np.array(metrics.precision_recall_fscore_support(df.filter(sel).x.values, df.filter(sel).y.values, average=average)[:3])
            np.testing.assert_array_almost_equal(vaex_score[i], sklearn_score)


@pytest.mark.parametrize('df', [df_binary, df_multi_class, df_multi_class_strings])
@pytest.mark.parametrize('selection', [None, [None, 'z==0', 'z==1']])
def test_matthews_correlation_coefficient(df, selection):
    if selection is None:
        vaex_score = df.ml.metrics.matthews_correlation_coefficient(df.x, df.y)
        sklearn_score = metrics.matthews_corrcoef(df.x.values, df.y.values)
        assert vaex_score == sklearn_score
    else:
        vaex_score = df.ml.metrics.matthews_correlation_coefficient(df.x, df.y, selection=selection)
        for i, sel in enumerate(selection):
            sklearn_score = metrics.matthews_corrcoef(df.filter(sel).x.values, df.filter(sel).y.values)
            assert vaex_score[i] == sklearn_score


@pytest.mark.parametrize('df', [df_binary, df_multi_class])
@pytest.mark.parametrize('selection', [None, [None, 'z==0']])
def test_confusion_matrix(df, selection):
    if selection is None:
        vaex_result = df.ml.metrics.confusion_matrix(df.x, df.y)
        sklearn_result = metrics.confusion_matrix(df.x.values, df.y.values)
        np.testing.assert_array_equal(vaex_result, sklearn_result)
    else:
        vaex_result = df.ml.metrics.confusion_matrix(df.x, df.y, selection=selection)
        for i, sel in enumerate(selection):
            sklearn_result = metrics.confusion_matrix(df.filter(sel).x.values, df.filter(sel).y.values)
            np.testing.assert_array_equal(vaex_result[i], sklearn_result)


@pytest.mark.parametrize('selection', [None, [None, 'class_==0', 'class_==1']])
def test_mean_absolute_error(selection):
    df = vaex.datasets.iris()
    if selection is None:
        vaex_result = df.ml.metrics.mean_absolute_error(df.petal_width, df.petal_length)
        sklearn_result = metrics.mean_absolute_error(df.petal_width.values, df.petal_length.values)
        assert vaex_result == sklearn_result
    else:
        vaex_result = df.ml.metrics.mean_absolute_error(df.petal_width, df.petal_length, selection=selection)
        for i, sel in enumerate(selection):
            sklearn_result = metrics.mean_absolute_error(df.filter(sel).petal_width.values, df.filter(sel).petal_length.values)
            np.testing.assert_almost_equal(vaex_result[i], sklearn_result)


@pytest.mark.parametrize('selection', [None, [None, 'class_==0', 'class_==1']])
def test_mean_squared_error(selection):
    df = vaex.datasets.iris()
    if selection is None:
        vaex_result = df.ml.metrics.mean_squared_error(df.petal_width, df.petal_length)
        sklearn_result = metrics.mean_squared_error(df.petal_width.values, df.petal_length.values)
        assert vaex_result == sklearn_result
    else:
        vaex_result = df.ml.metrics.mean_squared_error(df.petal_width, df.petal_length, selection=selection)
        for i, sel in enumerate(selection):
            sklearn_result = metrics.mean_squared_error(df.filter(sel).petal_width.values, df.filter(sel).petal_length.values)
            np.testing.assert_almost_equal(vaex_result[i], sklearn_result)


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
    expected_result = '\n        Classification report:\n\n        Accuracy:  0.778\n        Precision: 0.767\n        Recall:    0.722\n        F1:        0.73\n        '
    assert report == expected_result

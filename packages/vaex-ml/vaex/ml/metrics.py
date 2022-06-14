import warnings

import numpy as np

import vaex


def ensure_string_arguments(*args):
    result = []
    for arg in args:
        result.append(vaex.utils._ensure_string_from_expression(arg))
    return result


def _prf_divide(numerator, denominator, metric, modifier, average, warn_for, zero_division="warn"):
    '''Performs division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to 0 or 1 (according to ``zero_division``).
    Plus, if ``zero_division != "warn"`` raises a warning.

    The metric, modifier and average arguments are used only for determining an appropriate warning.

    Note: this function was forked from the https://github.com/scikit-learn/scikit-learn/ project
    and was originally published under BSD-3 license, which is included in packages/vaex-ml/SCIKIT_LEARN_LICENSE.txt
    '''
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # if ``zero_division=1``, set those with denominator == 0 equal to 1
    result[mask] = 0.0 if zero_division in ["warn", 0] else 1.0

    # the user will be removing warnings if zero_division is set to something
    # different than its default value. If we are computing only f-score
    # the warning will be raised only if precision and recall are ill-defined
    if zero_division != "warn" or metric not in warn_for:
        return result

    # build appropriate warning
    # E.g. "Precision and F-score are ill-defined and being set to 0.0 in
    # labels with no predicted samples. Use ``zero_division`` parameter to
    # control this behavior."

    if metric in warn_for and "f-score" in warn_for:
        msg_start = "{0} and F-score are".format(metric.title())
    elif metric in warn_for:
        msg_start = "{0} is".format(metric.title())
    elif "f-score" in warn_for:
        msg_start = "F-score is"
    else:
        return result

    _warn_prf(average, modifier, msg_start, len(result))

    return result


def _warn_prf(average, modifier, msg_start, result_size):
    '''
    Note: this function was forked from the https://github.com/scikit-learn/scikit-learn/ project
    and was originally published under BSD-3 license, which is included in packages/vaex-ml/SCIKIT_LEARN_LICENSE.txt
    '''
    axis0, axis1 = "sample", "label"
    if average == "samples":
        axis0, axis1 = axis1, axis0
    msg = (
        "{0} ill-defined and being set to 0.0 {{0}} "
        "no {1} {2}s. Use `zero_division` parameter to control"
        " this behavior.".format(msg_start, modifier, axis0)
    )
    if result_size == 1:
        msg = msg.format("due to")
    else:
        msg = msg.format("in {0}s with".format(axis1))
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


class UndefinedMetricWarning(UserWarning):
    '''Warning used when the metric is invalid

    (this function is taken verbatim from scikit-learn)
    '''


class DataFrameAccessorMetrics():
    '''Common metrics for evaluating machine learning tasks.

    This DataFrame Accessor contains a number of common machine learning evaluation metrics.
    The idea is that the metrics can be evaluated out-of-core, and without the need to materialize the target and predicted columns.

    See https://vaex.io/docs/api.html#metrics for a list of supported evaluation metrics.
    '''
    def __init__(self, ml):
        self.ml = ml
        self.df = self.ml.df

    @vaex.docstrings.docsubst
    def accuracy_score(self, y_true, y_pred, selection=None, array_type='python'):
        '''
        Calculates the accuracy classification score.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param selection: {selection}
        :param array_type: {array_type}
        :returns: The accuracy score.

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0], y_pred=[1, 0, 0, 1, 1])
        >>> df.ml.metrics.accuracy_score(df.y_true, df.y_pred)
          0.6
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        acc = (self.df[y_true] == self.df[y_pred]).sum(selection=selection) / self.df.count(selection=selection)
        if vaex.utils._issequence(acc):
            return vaex.array_types.convert(acc, type=array_type)
        else:
            return acc

    @vaex.docstrings.docsubst
    def confusion_matrix(self, y_true, y_pred, selection=None, array_type=None):
        '''
        Docstrings
        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param selection: {selection}
        :param array_type: {array_type}
        :returns: The confusion matrix

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.confusion_matrix(df.y_true, df.y_pred)
          array([[1, 1],
                 [1, 3]]
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        df = self.df.copy()  # To not modify the original DataFrame

        if df.is_category(y_true) is not True:
            df = df.ordinal_encode(y_true)
        if df.is_category(y_pred) is not True:
            df = df.ordinal_encode(y_pred)

        return df.count(binby=(y_true, y_pred), selection=selection, array_type=array_type)

    @vaex.docstrings.docsubst
    def precision_recall_fscore(self, y_true, y_pred, average='binary', selection=None, array_type=None):
        '''Calculates the precision, recall and f1 score for a classification problem.

        These metrics are defined as follows:
        - precision = tp / (tp + fp)
        - recall = tp / (tp + fn)
        - f1 = tp / (tp + 0.5 * (fp + fn))
        where "tp" are true positives, "fp" are false positives, and "fn" are false negatives.

        For a binary classification problem, `average` should be set to "binary".
        In this case it is assumed that the input data is encoded in 0 and 1 integers, where the class of importance is labeled as 1.

        For multiclass classification problems, `average` should be set to "macro".
        The "macro" average is the unweighted mean of a metric for each label.
        For multiclass problems the data can be ordinal encoded, but class names are also supported.

        :y_true: {expression_one}
        :y_pred: {expression_one}
        :average: Should be either 'binary' or 'macro'.
        :selection: {selection}
        :array_type: {array_type}
        :returns: The precision, recall and f1 score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.precision_score(df.y_true, df.y_pred)
          (0.75, 0.75, 0.75)
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        assert average in ['binary', 'macro']

        C = self.confusion_matrix(y_true=y_true, y_pred=y_pred, array_type='numpy', selection=selection)

        if average == 'binary':
            if (len(C.shape) == 2) & (C.shape == (2, 2)):
                Cdiag = np.diag(C)
                precision = _prf_divide(Cdiag, np.sum(C, axis=0), 'precision', 'predicted', average, 'precision')[1]
                recall = _prf_divide(Cdiag, np.sum(C, axis=1), 'recall', 'predicted', average, 'recall')[1]
                f1 = _prf_divide(vaex.array_types.to_numpy(2 * precision * recall),
                                 vaex.array_types.to_numpy(precision + recall),
                                 'f1', 'predicted', average, 'f1').item()
            elif (len(C.shape) == 3) & (C.shape[1:] == (2, 2)):
                Cdiag = np.array([np.diag(i) for i in C])
                precision = _prf_divide(Cdiag, np.array([np.sum(i, axis=0) for i in C]), 'precision', 'predicted', average, 'precision')[:, 1]
                recall = _prf_divide(Cdiag, np.array([np.sum(i, axis=1) for i in C]), 'recall', 'predicted', average, 'recall')[:, 1]
                f1 = _prf_divide(2 * precision * recall, precision + recall, 'f1', 'predicted', average, 'f1')
            else:
                raise ValueError('Cannot calculate metrics for `average="binary"`.')

        else:
            if len(C.shape) == 2:
                Cdiag = np.diag(C)
                precision_array = _prf_divide(Cdiag, np.sum(C, axis=0), 'precision', 'predicted', average, 'precision')
                recall_array = _prf_divide(Cdiag, np.sum(C, axis=1), 'recall', 'predicted', average, 'recall')
                f1_array = _prf_divide(vaex.array_types.to_numpy(2 * precision_array * recall_array),
                                       vaex.array_types.to_numpy(precision_array + recall_array), 'f1', 'predicted', average, 'f1')
                precision = precision_array.mean()
                recall = recall_array.mean()
                f1 = f1_array.mean()
            if len(C.shape) == 3:
                Cdiag = np.array([np.diag(i) for i in C])
                precision_array = _prf_divide(Cdiag, np.array([np.sum(i, axis=0) for i in C]), 'precision', 'predicted', average, 'precision')
                recall_array = _prf_divide(Cdiag, np.array([np.sum(i, axis=1) for i in C]), 'recall', 'predicted', average, 'recall')
                f1_array = _prf_divide(2 * precision_array * recall_array, precision_array + recall_array, 'f1', 'predicted', average, 'f1')
                precision = precision_array.mean(axis=1)
                recall = recall_array.mean(axis=1)
                f1 = f1_array.mean(axis=1)

        if vaex.utils._issequence(precision):
            return (vaex.array_types.convert(precision, type=array_type),
                    vaex.array_types.convert(recall, type=array_type),
                    vaex.array_types.convert(f1, type=array_type))
        return precision, recall, f1

    @vaex.docstrings.docsubst
    def precision_score(self, y_true, y_pred, average='binary', selection=None, array_type=None):
        '''Calculates the precision classification score.

        For a binary classification problem, `average` should be set to "binary".
        In this case it is assumed that the input data is encoded in 0 and 1 integers, where the class of importance is labeled as 1.

        For multiclass classification problems, `average` should be set to "macro".
        The "macro" average is the unweighted mean of a metric for each label.
        For multiclass problems the data can be ordinal encoded, but class names are also supported.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param average: Should be either 'binary' or 'macro'.
        :param selection: {selection}
        :param array_type: {array_type}
        :returns: The precision score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.precision_score(df.y_true, df.y_pred)
          0.75
        '''

        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        precision, _, _ = self.precision_recall_fscore(y_true, y_pred, average=average, selection=selection, array_type=array_type)
        return precision

    @vaex.docstrings.docsubst
    def recall_score(self, y_true, y_pred, average='binary', selection=None, array_type=None):
        '''
        Calculates the recall classification score.

        For a binary classification problem, `average` should be set to "binary".
        In this case it is assumed that the input data is encoded in 0 and 1 integers, where the class of importance is labeled as 1.

        For multiclass classification problems, `average` should be set to "macro".
        The "macro" average is the unweighted mean of a metric for each label.
        For multiclass problems the data can be ordinal encoded, but class names are also supported.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param average: Should be either 'binary' or 'macro'.
        :param selection: {selection}
        :param array_type: {array_type}
        :returns: The recall score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.recall_score(df.y_true, df.y_pred)
          0.75
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        _, recall, _ = self.precision_recall_fscore(y_true, y_pred, average=average, selection=selection, array_type=array_type)
        return recall

    def f1_score(self, y_true, y_pred, average='binary', selection=None, array_type=None):
        '''Calculates the F1 score.

        This is the harmonic average between the precision and the recall.

        For a binary classification problem, `average` should be set to "binary".
        In this case it is assumed that the input data is encoded in 0 and 1 integers, where the class of importance is labeled as 1.

        For multiclass classification problems, `average` should be set to "macro".
        The "macro" average is the unweighted mean of a metric for each label.
        For multiclass problems the data can be ordinal encoded, but class names are also supported.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param average: Should be either 'binary' or 'macro'.
        :param selection: {selection}
        :param array_type: {array_type}
        :returns: The recall score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.recall_score(df.y_true, df.y_pred)
          0.75
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        _, _, f1 = self.precision_recall_fscore(y_true, y_pred, average=average, selection=selection, array_type=array_type)
        return f1

    def matthews_correlation_coefficient(self, y_true, y_pred, selection=None, array_type=None):
        '''Calculates the Matthews correlation coefficient.

        This metric can be used for both binary and multiclass classification problems.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param selection: {selection}
        :returns: The Matthews correlation coefficient.

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.matthews_correlation_coefficient(df.y_true, df.y_pred)
          0.25
        '''
        C = self.confusion_matrix(y_true=y_true, y_pred=y_pred, selection=selection, array_type='numpy')
        if len(C.shape) == 2:
            # This is from scikit-learn
            t_sum = C.sum(axis=1, dtype=np.float64)
            p_sum = C.sum(axis=0, dtype=np.float64)
            n_correct = np.trace(C, dtype=np.float64)
            n_samples = p_sum.sum()
            cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
            cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
            cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)

            if cov_ypyp * cov_ytyt == 0:
                return 0.0
            else:
                return cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
        else:
            t_sum = np.array([i.sum(axis=1, dtype=np.float64) for i in C])
            p_sum = np.array([i.sum(axis=0, dtype=np.float64) for i in C])
            n_correct = np.array([np.trace(i, dtype=np.float64) for i in C])
            n_samples =p_sum.sum(axis=1)
            cov_ytyp = n_correct * n_samples - np.array([np.dot(i, j) for i, j in zip(t_sum, p_sum)])
            cov_ypyp = n_samples ** 2 - np.array([np.dot(i, i) for i in p_sum])
            cov_ytyt = n_samples ** 2 - np.array([np.dot(i, i) for i in t_sum])
            mcc = _prf_divide(cov_ytyp, np.sqrt(cov_ytyt * cov_ypyp), metric='MCC', modifier='predicted', average='n/a', warn_for=[None])

            if vaex.utils._issequence(mcc):
                return vaex.array_types.convert(mcc, type=array_type)
            return mcc


    @vaex.docstrings.docsubst
    def classification_report(self, y_true, y_pred, average='binary', decimals=3):
        '''Returns a text report showing the main classification metrics

        The accuracy, precision, recall, and F1-score are shown.

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> report = df.ml.metrics.classification_report(df.y_true, df.y_pred)
        >>> print(report)
        >>> print(report)
            Classification report:

            Accuracy:  0.667
            Precision: 0.75
            Recall:    0.75
            F1:        0.75
        '''
        accuracy_score = self.accuracy_score(y_true=y_true, y_pred=y_pred)
        precision_score, recall_score, f1_score = self.precision_recall_fscore(y_true=y_true, y_pred=y_pred, average=average)
        report = f'''
        Classification report:

        Accuracy:  {accuracy_score:.{decimals}}
        Precision: {precision_score:.{decimals}}
        Recall:    {recall_score:.{decimals}}
        F1:        {f1_score:.{decimals}}
        '''
        return report

    @vaex.docstrings.docsubst
    def mean_absolute_error(self, y_true, y_pred, selection=None, array_type='python'):
        '''Calculate the mean absolute error.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param selection: {selection}
        :param str array_type: {array_type}
        :returns: The mean absolute error

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.datasets.iris()
        >>> df.ml.metrics.mean_absolute_error(df.sepal_length, df.petal_length)
          2.0846666666666667
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        score = (np.abs(self.df[y_true] - self.df[y_pred])).mean(selection=selection)

        if vaex.utils._issequence(selection):
            return vaex.array_types.convert(score, type=array_type)
        else:
            return score.item()

    @vaex.docstrings.docsubst
    def mean_squared_error(self, y_true, y_pred, selection=None, array_type='python'):
        '''Calculates the mean squared error.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param selection: {selection}
        :param str array_type: {array_type}
        :returns: The mean squared error

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.datasets.iris()
        >>> df.ml.metrics.mean_squared_error(df.sepal_length, df.petal_length)
          5.589000000000001
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        score = ((self.df[y_true] - self.df[y_pred])**2).mean(selection=selection)

        if vaex.utils._issequence(selection):
            return vaex.array_types.convert(score, type=array_type)
        else:
            return score.item()

    @vaex.docstrings.docsubst
    def r2_score(self, y_true, y_pred):
        '''Calculates the R**2 (coefficient of determination) regression score function.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param selection: {selection}
        :param str array_type: {array_type}
        :returns: The R**2 score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.datasets.iris()
        >>> df.ml.metrics.r2_score(df.sepal_length, df.petal_length)
          -7.205575765485069
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)

        numerator = ((self.df[y_true] - self.df[y_pred])**2).sum()
        denominator = ((self.df[y_true] - self.df[y_true].mean())**2).sum()
        return 1 - _prf_divide(numerator, denominator, metric='R2', modifier='predicted', average='n/a', warn_for=[None])

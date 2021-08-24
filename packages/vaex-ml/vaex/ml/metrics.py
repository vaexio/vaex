import numpy as np

import vaex


def ensure_string_arguments(*args):
    result = []
    for arg in args:
        result.append(vaex.utils._ensure_string_from_expression(arg))
    return result


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
    def accuracy_score(self, y_true, y_pred):
        '''
        Calculates the accuracy classification score.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :returns: The accuracy score.

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0], y_pred=[1, 0, 0, 1, 1])
        >>> df.ml.metrics.accuracy_score(df.y_true, df.y_pred)
          0.6
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        return (self.df[y_true] == self.df[y_pred]).sum() / len(self.df)

    @vaex.docstrings.docsubst
    def confusion_matrix(self, y_true, y_pred, array_type=None):
        '''
        Docstrings
        :param y_true: {expression_one}
        :param y_pred: {expression_one}
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

        return df.count(binby=(y_pred, y_true), array_type=array_type).T

    @vaex.docstrings.docsubst
    def precision_recall_fscore(self, y_true, y_pred, average='binary'):
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

        Example:
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        assert average in ['binary', 'macro']

        if average == 'binary':
            assert set(self.df[y_true].unique()) == {0, 1}, '`y_true` must be encoded in 1s and 0s'
            assert set(self.df[y_pred].unique()) == {0, 1}, '`y_pred` must be encoded in 1s and 0s'
            selections = [f'({y_true}==1) & ({y_pred}==1)', f'{y_pred}==1', f'{y_true}==1']
            count_y_true_y_pred, count_y_pred, count_y_true = self.df.count(selection=selections)
            precision = count_y_true_y_pred / count_y_pred
            recall = count_y_true_y_pred / count_y_true
            f1 = 2 * (precision * recall) / (precision + recall)

        else:
            C = self.confusion_matrix(y_true=y_true, y_pred=y_pred, array_type='numpy')
            precision_array = (np.diag(C) / np.sum(C, axis=0))
            recall_array = (np.diag(C) / np.sum(C, axis=1))
            f1_array = 2 * (precision_array * recall_array) / (precision_array + recall_array)
            precision = precision_array.mean()
            recall = recall_array.mean()
            f1 = f1_array.mean()

        return precision, recall, f1

    @vaex.docstrings.docsubst
    def precision_score(self, y_true, y_pred, average='binary'):
        '''Calculates the precision classification score.

        For a binary classification problem, `average` should be set to "binary".
        In this case it is assumed that the input data is encoded in 0 and 1 integers, where the class of importance is labeled as 1.

        For multiclass classification problems, `average` should be set to "macro".
        The "macro" average is the unweighted mean of a metric for each label.
        For multiclass problems the data can be ordinal encoded, but class names are also supported.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :param average: Should be either 'binary' or 'macro'.
        :returns: The precision score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.precision_score(df.y_true, df.y_pred)
          0.75
        '''

        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        precision, _, _ = self.precision_recall_fscore(y_true, y_pred, average=average)
        return precision

    @vaex.docstrings.docsubst
    def recall_score(self, y_true, y_pred, average='binary'):
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
        :returns: The recall score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.recall_score(df.y_true, df.y_pred)
          0.75
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        _, recall, _ = self.precision_recall_fscore(y_true, y_pred, average=average)
        return recall

    def f1_score(self, y_true, y_pred, average='binary'):
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
        :returns: The recall score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.recall_score(df.y_true, df.y_pred)
          0.75
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        _, _, f1 = self.precision_recall_fscore(y_true, y_pred, average=average)
        return f1

    def matthews_correlation_coefficient(self, y_true, y_pred):
        '''Calculates the Matthews correlation coefficient.

        This metric can be used for both binary and multiclass classification problems.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :returns: The Matthews correlation coefficient.

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.from_arrays(y_true=[1, 1, 0, 1, 0, 1], y_pred=[1, 0, 0, 1, 1, 1])
        >>> df.ml.metrics.matthews_correlation_coefficient(df.y_true, df.y_pred)
          0.25
        '''
        C = self.confusion_matrix(y_true=y_true, y_pred=y_pred)
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
    def mean_absolute_error(self, y_true, y_pred):
        '''Calculate the mean absolute error.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :returns: The mean absolute error

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.ml.datasets.load_iris()
        >>> df.ml.metrics.mean_absolute_error(df.sepal_length, df.petal_length)
          2.0846666666666667
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        return (np.abs(self.df[y_true] - self.df[y_pred])).mean().item()

    @vaex.docstrings.docsubst
    def mean_squared_error(self, y_true, y_pred):
        '''Calculates the mean squared error.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :returns: The mean squared error

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.ml.datasets.load_iris()
        >>> df.ml.metrics.mean_squared_error(df.sepal_length, df.petal_length)
          5.589000000000001
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)
        return ((self.df[y_true] - self.df[y_pred])**2).mean().item()

    @vaex.docstrings.docsubst
    def r2_score(self, y_true, y_pred):
        '''Calculates the R**2 (coefficient of determination) regression score function.

        :param y_true: {expression_one}
        :param y_pred: {expression_one}
        :returns: The R**2 score

        Example:

        >>> import vaex
        >>> import vaex.ml.metrics
        >>> df = vaex.ml.datasets.load_iris()
        >>> df.ml.metrics.r2_score(df.sepal_length, df.petal_length)
          -7.205575765485069
        '''
        y_true, y_pred = ensure_string_arguments(y_true, y_pred)

        numerator = ((self.df[y_true] - self.df[y_pred])**2).sum()
        denominator = ((self.df[y_true] - self.df[y_true].mean())**2).sum()
        if denominator == 0:
            return 0.0
        else:
            return 1 - (numerator / denominator)

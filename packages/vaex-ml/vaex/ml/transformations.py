import vaex.dataframe
from vaex.serialize import register
import numpy as np
from . import generate
from .state import HasState
import traitlets
from vaex.utils import _ensure_strings_from_expressions

help_features = 'List of features to transform.'
help_prefix = 'Prefix for the names of the transformed features.'


def dot_product(a, b):
    products = ['%s * %s' % (ai, bi) for ai, bi in zip(a, b)]
    return ' + '.join(products)


@register
class StateTransfer(HasState):
    state = traitlets.Dict()

    def transform(self, df):
        copy = df.copy()
        self.state = dict(self.state, active_range=[copy._index_start, copy._index_end])
        copy.state_set(self.state)
        return copy


class Transformer(HasState):
    ''' Parent class for all of the transformers.
    '''
    features = traitlets.List(traitlets.Unicode(), help=help_features).tag(ui='SelectMultiple')

    def fit_transform(self, df):
        '''Fit and apply the transformer to the supplied DataFrame.

        :param df: A vaex DataFrame.

        :returns copy: A shallow copy of the DataFrame that includes the transformations.
        '''
        self.fit(df=df)
        return self.transform(df=df)


@register
@generate.register
class PCA(Transformer):
    '''Transform a set of features using a Principal Component Analysis.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10])
    >>> df
     #   x   y
     0   2   -2
     1   5   3
     2   7   0
     3   2   0
     4   15  10
    >>> pca = vaex.ml.PCA(n_components=2, features=['x', 'y'])
    >>> pca.fit_transform(df)
     #    x    y       PCA_0      PCA_1
     0    2   -2    5.92532    0.413011
     1    5    3    0.380494  -1.39112
     2    7    0    0.840049   2.18502
     3    2    0    4.61287   -1.09612
     4   15   10  -11.7587    -0.110794

    '''
    # title = traitlets.Unicode(default_value='PCA', read_only=True).tag(ui='HTML')
    n_components = traitlets.Int(help='Number of components to retain. If None, all the components will be retained.').tag(ui='IntText')
    prefix = traitlets.Unicode(default_value="PCA_", help=help_prefix)
    progress = traitlets.CBool(default_value=False, help='If True, display a progressbar of the PCA fitting process.').tag(ui='Checkbox')
    eigen_vectors_ = traitlets.List(traitlets.List(traitlets.CFloat()), help='The eigen vectors corresponding to each feature').tag(output=True)
    eigen_values_ = traitlets.List(traitlets.CFloat(), help='The eigen values that correspond to each feature.').tag(output=True)
    means_ = traitlets.List(traitlets.CFloat(), help='The mean of each feature').tag(output=True)

    @traitlets.default('n_components')
    def get_n_components_default(self):
        return len(self.features)

    def fit(self, df):
        '''Fit the PCA model to the DataFrame.

        :param df: A vaex DataFrame.
        '''
        self.n_components = self.n_components or len(self.features)
        assert self.n_components >= 2, 'At least two features are required.'
        assert self.n_components <= len(self.features), 'Can not have more components than features.'
        C = df.cov(self.features, progress=self.progress)
        eigen_values, eigen_vectors = np.linalg.eigh(C)
        indices = np.argsort(eigen_values)[::-1]
        self.means_ = df.mean(self.features, progress=self.progress).tolist()
        self.eigen_vectors_ = eigen_vectors[:, indices].tolist()
        self.eigen_values_ = eigen_values[indices].tolist()

    def transform(self, df, n_components=None):
        '''Apply the PCA transformation to the DataFrame.

        :param df: A vaex DataFrame.
        :param n_components: The number of PCA components to retain.

        :return copy: A shallow copy of the DataFrame that includes the PCA components.
        :rtype: DataFrame
        '''
        n_components = n_components or self.n_components
        copy = df.copy()
        name_prefix_offset = 0
        eigen_vectors = np.array(self.eigen_vectors_)
        while self.prefix + str(name_prefix_offset) in copy.get_column_names(virtual=True, strings=True):
            name_prefix_offset += 1

        expressions = [copy[feature]-mean for feature, mean in zip(self.features, self.means_)]
        for i in range(n_components):
            v = eigen_vectors[:, i]
            expr = dot_product(expressions, v)
            name = self.prefix + str(i + name_prefix_offset)
            copy[name] = expr
        return copy


@register
@generate.register
class LabelEncoder(Transformer):
    '''Encode categorical columns with integer values between 0 and num_classes-1.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(color=['red', 'green', 'green', 'blue', 'red'])
    >>> df
     #  color
     0  red
     1  green
     2  green
     3  blue
     4  red
    >>> encoder = vaex.ml.LabelEncoder(features=['color'])
    >>> encoder.fit_transform(df)
     #  color      label_encoded_color
     0  red                          2
     1  green                        1
     2  green                        1
     3  blue                         0
     4  red                          2
    '''
    # title = traitlets.Unicode(default_value='Label Encoder', read_only=True).tag(ui='HTML')
    prefix = traitlets.Unicode(default_value='label_encoded_', help=help_prefix).tag(ui='Text')
    labels_ = traitlets.Dict(default_value={}, allow_none=True, help='The encoded labels of each feature.').tag(output=True)
    allow_unseen = traitlets.Bool(default_value=False, allow_none=False, help='If True, unseen values will be \
                                  encoded with -1, otherwise an error is raised').tag(ui='Checkbox')

    def fit(self, df):
        '''Fit LabelEncoder to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        for feature in self.features:
            labels = df[feature].unique().tolist()
            self.labels_[feature] = dict(zip(labels, np.arange(len(labels))))

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted LabelEncoder.

        :param df: A vaex DataFrame.

        Returns:
        :return copy: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''

        default_value = None
        if self.allow_unseen:
            default_value = -1

        copy = df.copy()
        for feature in self.features:
            name = self.prefix + feature
            copy[name] = copy[feature].map(mapper=self.labels_[feature], default_value=default_value)

        return copy


@register
@generate.register
class OneHotEncoder(Transformer):
    '''Encode categorical columns according ot the One-Hot scheme.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(color=['red', 'green', 'green', 'blue', 'red'])
    >>> df
     #  color
     0  red
     1  green
     2  green
     3  blue
     4  red
    >>> encoder = vaex.ml.OneHotEncoder(features=['color'])
    >>> encoder.fit_transform(df)
     #  color      color_blue    color_green    color_red
     0  red                 0              0            1
     1  green               0              1            0
     2  green               0              1            0
     3  blue                1              0            0
     4  red                 0              0            1
    '''

    # title = Unicode(default_value='One-Hot Encoder', read_only=True).tag(ui='HTML')
    prefix = traitlets.Unicode(default_value='', help=help_prefix).tag(ui='Text')
    one = traitlets.Any(1, help='Value to encode when a category is present.')
    zero = traitlets.Any(0, help='Value to encode when category is absent.')
    uniques_ = traitlets.List(traitlets.List(), help='The unique elements found in each feature.').tag(output=True)

    def fit(self, df):
        '''Fit OneHotEncoder to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        uniques = []
        for i in self.features:
            expression = _ensure_strings_from_expressions(i)
            unique = df.unique(expression)
            unique = np.sort(unique)  # this can/should be optimized with @delay
            uniques.append(unique.tolist())
        self.uniques_ = uniques

    def transform(self, df):
        '''Transform a DataFrame with a fitted OneHotEncoder.

        :param df: A vaex DataFrame.
        :return: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''
        copy = df.copy()
        # for each feature, add a virtual column for each unique entry
        for i, feature in enumerate(self.features):
            for j, value in enumerate(self.uniques_[i]):
                column_name = self.prefix + feature + '_' + str(value)
                copy.add_virtual_column(column_name, 'where({feature} == {value}, {one}, {zero})'.format(
                                        feature=feature, value=repr(value), one=self.one, zero=self.zero))
        return copy


@register
@generate.register
class FrequencyEncoder(Transformer):
    '''Encode categorical columns by the frequency of their respective samples.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(color=['red', 'green', 'green', 'blue', 'red', 'green'])
    >>> df
     #  color
     0  red
     1  green
     2  green
     3  blue
     4  red
    >>> encoder = vaex.ml.FrequencyEncoder(features=['color'])
    >>> encoder.fit_transform(df)
     #  color      frequency_encoded_color
     0  red                       0.333333
     1  green                     0.5
     2  green                     0.5
     3  blue                      0.166667
     4  red                       0.333333
     5  green                     0.5
    '''
    prefix = traitlets.Unicode(default_value='frequency_encoded_', help=help_prefix).tag(ui='Text')
    unseen = traitlets.Enum(values=['zero', 'nan'], default_value='nan', help='Strategy to deal with unseen values.')
    mappings_ = traitlets.Dict()

    def fit(self, df):
        '''Fit FrequencyEncoder to the DataFrame.

        :param df: A vaex DataFrame.
        '''
        # number of samples
        nsamples = len(df)

        # Encoding
        for feature in self.features:
            self.mappings_[feature] = dict(df[feature].value_counts() / nsamples)

    def transform(self, df):
        '''Transform a DataFrame with a fitted FrequencyEncoder.

        :param df: A vaex DataFrame.
        :return: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''
        copy = df.copy()
        default_value = {'zero': 0., 'nan': np.nan}[self.unseen]
        for feature in self.features:
            name = self.prefix + feature
            expression = copy[feature].map(self.mappings_[feature], nan_value=np.nan, missing_value=np.nan, default_value=default_value, allow_missing=True)

            copy[name] = expression
        return copy


@register
@generate.register
class StandardScaler(Transformer):
    '''Standardize features by removing thir mean and scaling them to unit variance.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10])
    >>> df
     #    x    y
     0    2   -2
     1    5    3
     2    7    0
     3    2    0
     4   15   10
    >>> scaler = vaex.ml.StandardScaler(features=['x', 'y'])
    >>> scaler.fit_transform(df)
     #    x    y    standard_scaled_x    standard_scaled_y
     0    2   -2            -0.876523            -0.996616
     1    5    3            -0.250435             0.189832
     2    7    0             0.166957            -0.522037
     3    2    0            -0.876523            -0.522037
     4   15   10             1.83652              1.85086
    '''
    # title = Unicode(default_value='Standard Scaler', read_only=True).tag(ui='HTML')
    prefix = traitlets.Unicode(default_value="standard_scaled_", help=help_prefix).tag(ui='Text')
    with_mean = traitlets.CBool(default_value=True, help='If True, remove the mean from each feature.').tag(ui='Checkbox')
    with_std = traitlets.CBool(default_value=True, help='If True, scale each feature to unit variance.').tag(ui='Checkbox')
    mean_ = traitlets.List(traitlets.CFloat(), help='The mean of each feature').tag(output=True)
    std_ = traitlets.List(traitlets.CFloat(), help='The standard deviation of each feature.').tag(output=True)

    def fit(self, df):
        '''
        Fit StandardScaler to the DataFrame.

        :param df: A vaex DataFrame.
        '''
        mean = df.mean(self.features, delay=True)
        std = df.std(self.features, delay=True)

        @vaex.delayed
        def assign(mean, std):
            self.mean_ = mean.tolist()
            self.std_ = std.tolist()

        assign(mean, std)
        df.execute()

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted StandardScaler.

        :param df: A vaex DataFrame.

        :returns copy: a shallow copy of the DataFrame that includes the scaled features.
        :rtype: DataFrame
        '''

        copy = df.copy()
        for i in range(len(self.features)):
            name = self.prefix+self.features[i]
            expression = copy[self.features[i]]
            if self.with_mean:
                expression = expression - self.mean_[i]
            if self.with_std:
                expression = expression / self.std_[i]
            copy[name] = expression
        return copy


@register
@generate.register
class MinMaxScaler(Transformer):
    '''Will scale a set of features to a given range.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10])
    >>> df
     #    x    y
     0    2   -2
     1    5    3
     2    7    0
     3    2    0
     4   15   10
    >>> scaler = vaex.ml.MinMaxScaler(features=['x', 'y'])
    >>> scaler.fit_transform(df)
     #    x    y    minmax_scaled_x    minmax_scaled_y
     0    2   -2           0                  0
     1    5    3           0.230769           0.416667
     2    7    0           0.384615           0.166667
     3    2    0           0                  0.166667
     4   15   10           1                  1
    '''
    # title = Unicode(default_value='MinMax Scaler', read_only=True).tag(ui='HTML')
    feature_range = traitlets.Tuple(default_value=(0, 1), help='The range the features are scaled to.').tag().tag(ui='FloatRangeSlider')
    prefix = traitlets.Unicode(default_value="minmax_scaled_", help=help_prefix).tag(ui='Text')
    fmax_ = traitlets.List(traitlets.CFloat(), help='The minimum value of a feature.').tag(output=True)
    fmin_ = traitlets.List(traitlets.CFloat(), help='The maximum value of a feature.').tag(output=True)

    def fit(self, df):
        '''
        Fit MinMaxScaler to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        assert len(self.feature_range) == 2, 'feature_range must have 2 elements only'
        minmax = df.minmax(self.features)
        self.fmin_ = minmax[:, 0].tolist()
        self.fmax_ = minmax[:, 1].tolist()

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted MinMaxScaler.

        :param df: A vaex DataFrame.

        :return copy: a shallow copy of the DataFrame that includes the scaled features.
        :rtype: DataFrame
        '''

        copy = df.copy()

        for i in range(len(self.features)):
            name = self.prefix + self.features[i]
            a = self.feature_range[0]
            b = self.feature_range[1]
            expr = copy[self.features[i]]
            expr = (b-a)*(expr-self.fmin_[i])/(self.fmax_[i]-self.fmin_[i]) + a
            copy[name] = expr
        return copy


@register
@generate.register
class MaxAbsScaler(Transformer):
    ''' Scale features by their maximum absolute value.

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10])
    >>> df
     #    x    y
     0    2   -2
     1    5    3
     2    7    0
     3    2    0
     4   15   10
    >>> scaler = vaex.ml.MaxAbsScaler(features=['x', 'y'])
    >>> scaler.fit_transform(df)
     #    x    y    absmax_scaled_x    absmax_scaled_y
     0    2   -2           0.133333               -0.2
     1    5    3           0.333333                0.3
     2    7    0           0.466667                0
     3    2    0           0.133333                0
     4   15   10           1                       1
    '''
    prefix = traitlets.Unicode(default_value="absmax_scaled_", help=help_prefix).tag(ui='Text')
    absmax_ = traitlets.List(traitlets.CFloat(), help='Tha maximum absolute value of a feature.').tag(output=True)

    def fit(self, df):
        '''
        Fit MinMaxScaler to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        absmax = df.max(['abs(%s)' % k for k in self.features]).tolist()
        # Check if the absmax_ value is 0, in which case replace with 1
        self.absmax_ = [value if value != 0 else 1 for value in absmax]

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted MaxAbsScaler.

        :param df: A vaex DataFrame.

        :return copy: a shallow copy of the DataFrame that includes the scaled features.
        :rtype: DataFrame
        '''

        copy = df.copy()
        for i in range(len(self.features)):
            name = self.prefix + self.features[i]
            expr = copy[self.features[i]]
            expr = expr / self.absmax_[i]
            copy[name] = expr
        return copy


@register
@generate.register
class RobustScaler(Transformer):
    ''' The RobustScaler removes the median and scales the data according to a
    given percentile range. By default, the scaling is done between the 25th and
    the 75th percentile. Centering and scaling happens independently for each
    feature (column).

    Example:

    >>> import vaex
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10])
    >>> df
     #    x    y
     0    2   -2
     1    5    3
     2    7    0
     3    2    0
     4   15   10
    >>> scaler = vaex.ml.MaxAbsScaler(features=['x', 'y'])
    >>> scaler.fit_transform(df)
     #    x    y    robust_scaled_x    robust_scaled_y
     0    2   -2       -0.333686             -0.266302
     1    5    3       -0.000596934           0.399453
     2    7    0        0.221462              0
     3    2    0       -0.333686              0
     4   15   10        1.1097                1.33151
    '''
    with_centering = traitlets.CBool(default_value=True, help='If True, remove the median.').tag(ui='Checkbox')
    with_scaling = traitlets.CBool(default_value=True, help='If True, scale each feature between the specified percentile range.').tag(ui='Checkbox')
    percentile_range = traitlets.Tuple(default_value=(25, 75), help='The percentile range to which to scale each feature to.').tag().tag(ui='FloatRangeSlider')
    prefix = traitlets.Unicode(default_value="robust_scaled_", help=help_prefix).tag(ui='Text')
    center_ = traitlets.List(traitlets.CFloat(), default_value=None, help='The median of each feature.').tag(output=True)
    scale_ = traitlets.List(traitlets.CFloat(), default_value=None, help='The percentile range for each feature.').tag(output=True)

    def fit(self, df):
        '''
        Fit RobustScaler to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        # check the quantile range
        q_min, q_max = self.percentile_range
        if not 0 <= q_min <= q_max <= 100:
            raise ValueError('Invalid percentile range: %s' % (str(self.percentile_range)))

        if self.with_centering:
            self.center_ = df.percentile_approx(expression=self.features, percentage=50).tolist()

        if self.with_scaling:
            self.scale_ = (df.percentile_approx(expression=self.features, percentage=q_max) - df.percentile_approx(expression=self.features, percentage=q_min)).tolist()

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted RobustScaler.

        :param df: A vaex DataFrame.

        :returns copy: a shallow copy of the DataFrame that includes the scaled features.
        :rtype: DataFrame
        '''

        copy = df.copy()
        for i in range(len(self.features)):
            name = self.prefix+self.features[i]
            expr = copy[self.features[i]]
            if self.with_centering:
                expr = expr - self.center_[i]
            if self.with_scaling:
                expr = expr / self.scale_[i]
            copy[name] = expr
        return copy


@register
@generate.register
class CycleTransformer(Transformer):
    '''A strategy for transforming cyclical features (e.g. angles, time).

    Think of each feature as an angle of a unit circle in polar coordinates,
    and then and then obtaining the x and y coordinate projections,
    or the cos and sin components respectively.

    Suitable for a variaty of machine learning tasks.
    It preserves the cyclical continuity of the feature.
    Inspired by: http://blog.davidkaleko.com/feature-engineering-cyclical-features.html

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(days=[0, 1, 2, 3, 4, 5, 6])
    >>> cyctrans = vaex.ml.CycleTransformer(n=7, features=['days'])
    >>> cyctrans.fit_transform(df)
      #    days     days_x     days_y
      0       0   1          0
      1       1   0.62349    0.781831
      2       2  -0.222521   0.974928
      3       3  -0.900969   0.433884
      4       4  -0.900969  -0.433884
      5       5  -0.222521  -0.974928
      6       6   0.62349   -0.781831
    '''
    n = traitlets.CInt(allow_none=False, help='The number of elements in one cycle.')
    prefix_x = traitlets.Unicode(default_value="", help='Prefix for the x-component of the transformed features.').tag(ui='Text')
    prefix_y = traitlets.Unicode(default_value="", help='Prefix for the y-component of the transformed features.').tag(ui='Text')
    suffix_x = traitlets.Unicode(default_value="_x", help='Suffix for the x-component of the transformed features.').tag(ui='Text')
    suffix_y = traitlets.Unicode(default_value="_y", help='Suffix for the y-component of the transformed features.').tag(ui='Text')

    def fit(self, df):
        '''
        Fit a CycleTransformer to the DataFrame.

        This is a dummy method, as it is not needed for the transformation to be applied.

        :param df: A vaex DataFrame.
        '''
        pass

    def transform(self, df):
        '''
        Transform a DataFrame with a CycleTransformer.

        :param df: A vaex DataFrame.
        '''
        copy = df.copy()
        for feature in self.features:
            name_x = self.prefix_x + feature + self.suffix_x
            copy[name_x] = np.cos(2 * np.pi * copy[feature] / self.n)
            name_y = self.prefix_y + feature + self.suffix_y
            copy[name_y] = np.sin(2 * np.pi * copy[feature] / self.n)

        return copy


class BayesianTargetEncoder(Transformer):
    '''Encode categorical variables with a Bayesian Target Encoder.

    The categories are encoded by the mean of their target value,
    which is adjusted by the global mean value of the target variable
    using a Bayesian schema. For a larger `weight` value, the target
    encodings are smoothed toward the global mean, while for a
    `weight` of 0, the encodings are just the mean target value per
    class.

    Reference: https://www.wikiwand.com/en/Bayes_estimator#/Practical_example_of_Bayes_estimators

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(x=['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
    ...                       y=[1, 1, 1, 0, 0, 0, 0, 1])
    >>> target_encoder = vaex.ml.BayesianTargetEncoder(features=['x'], weight=4)
    >>> target_encoder.fit_transform(df, 'y')
      #  x      y    mean_encoded_x
      0  a      1             0.625
      1  a      1             0.625
      2  a      1             0.625
      3  a      0             0.625
      4  b      0             0.375
      5  b      0             0.375
      6  b      0             0.375
      7  b      1             0.375
    '''
    target = traitlets.Unicode(help='The name of the column containing the target variable.')
    weight = traitlets.CFloat(default_value=100, allow_none=False, help='Weight to be applied to the mean encodings (smoothing parameter).')
    prefix = traitlets.Unicode(default_value='mean_encoded_', help=help_prefix)
    unseen = traitlets.Enum(values=['zero', 'nan'], default_value='nan', help='Strategy to deal with unseen values.')
    mappings_ = traitlets.Dict()

    def fit(self, df):
        '''Fit a BayesianTargetEncoder to the DataFrame.

        :param df: A vaex DataFrame
        '''

        # The global target mean - used for the smoothing
        global_target_mean = df[self.target].mean().item()

        # TODO: we don't have delayed groupby yet, which could speed up the case with many features (1 pass over the data)
        for feature in self.features:
            agg = df.groupby(feature, agg={'count': vaex.agg.count(), 'mean': vaex.agg.mean(self.target)})
            agg['encoding'] = (agg['count'] * agg['mean'] + self.weight * global_target_mean) / (agg['count'] + self.weight)
            self.mappings_[feature] = {value[feature]: value['encoding'] for index, value in agg.iterrows()}

    def transform(self, df):
        '''Transform a DataFrame with a fitted BayesianTargetEncoder.

        :param df: A vaex DataFrame.
        :return: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''
        copy = df.copy()
        default_value = {'zero': 0., 'nan': np.nan}[self.unseen]
        for feature in self.features:
            name = self.prefix + feature
            copy[name] = copy[feature].map(self.mappings_[feature],
                                           nan_value=np.nan,
                                           missing_value=np.nan,
                                           default_value=default_value,
                                           allow_missing=True)
        return copy

class WeightOfEvidenceEncoder(Transformer):
    '''Encode categorical variables with a Weight of Evidence Encoder.

    Weight of Evidence measures how well a particular feature supports
    the given hypothesis (i.e. the target variable). With this
    encoder, each category in a categorical feature is encoded by its
    "strength" i.e. Weight of Evidence value. The target feature can be
    a boolean or numerical column, where True/1 is seen as 'Good', and
    False/0 is seen as 'Bad'

    Reference: https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(x=['a', 'a', 'b', 'b', 'b', 'c', 'c'],
    ...                       y=[1, 1, 0, 0, 1, 1, 0])
    >>> woe_encoder = vaex.ml.WeightOfEvidenceEncoder(target='y', features=['x'])
    >>> woe_encoder.fit_transform(df)
      #  x      y    mean_encoded_x
      0  a      1         13.8155
      1  a      1         13.8155
      2  b      0         -0.693147
      3  b      0         -0.693147
      4  b      1         -0.693147
      5  c      1          0
      6  c      0          0
    '''
    target = traitlets.Unicode(help='The name of the column containing the target variable.')
    prefix = traitlets.Unicode(default_value='woe_encoded_', help=help_prefix)
    unseen = traitlets.Enum(values=['zero', 'nan'], default_value='nan', help='Strategy to deal with unseen values.')
    epsilon = traitlets.Float(0.000001, help="Small value taken as minimum fot the negatives, to avoid a division by zero")
    mappings_ = traitlets.Dict()

    def fit(self, df):
        '''Fit a WeightOfEvidenceEncoder to the DataFrame.

        :param df: A vaex DataFrame
        '''
        values = df[self.target].unique(dropna=True)
        if not (
                (len(values) == 2 and (0 in values and 1 in values)) or \
                (len(values) == 1 and (0 in values or 1 in values)) or
                len(values) == 0 # all missing values
            ):
            raise ValueError("Target contains values different from True/1 and False/0: %r" % values)
        for feature in self.features:
            # Instead of counting the goods and bad, we divide by the count
            # which reduces to the mean
            agg = df.groupby(feature, agg={'positive': vaex.agg.mean(self.target)})
            agg['negative'] = 1 - agg.positive
            agg['negative'] = agg.func.where(agg['negative'] == 0, self.epsilon, agg['negative'])
            agg['woe'] = np.log(agg.positive/agg.negative)
            self.mappings_[feature] = {value[feature]: value['woe'] for index, value in agg.iterrows()}

    def transform(self, df):
        '''Transform a DataFrame with a fitted WeightOfEvidenceEncoder.

        :param df: A vaex DataFrame.
        :return: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''
        copy = df.copy()
        default_value = {'zero': 0., 'nan': np.nan}[self.unseen]
        for feature in self.features:
            name = self.prefix + feature
            copy[name] = copy[feature].map(self.mappings_[feature],
                                           nan_value=np.nan,
                                           missing_value=np.nan,
                                           default_value=default_value,
                                           allow_missing=True)

        return copy
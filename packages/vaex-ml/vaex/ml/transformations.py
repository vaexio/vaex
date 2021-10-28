import math
import numpy as np
import traitlets
import warnings

import vaex.dataframe
from vaex.ml import generate
from vaex.ml.state import HasState
from vaex.serialize import register
from vaex.utils import _ensure_strings_from_expressions


sklearn = vaex.utils.optional_import("sklearn", modules=[
    "sklearn.decomposition",
    "sklearn.random_projection"
])

help_features = 'List of features to transform.'
help_prefix = 'Prefix for the names of the transformed features.'


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
    >>> import vaex.ml
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
    n_components = traitlets.Int(default_value=None, allow_none=True, help='Number of components to retain. If None, all the components will be retained.').tag(ui='IntText')
    prefix = traitlets.Unicode(default_value="PCA_", help=help_prefix)
    whiten = traitlets.Bool(default_value=False, allow_none=False, help='If True perform whitening, i.e. remove the relative variance schale of the transformed components.')
    # progress = traitlets.Any(default_value=False, help='If True, display a progressbar of the PCA fitting process.').tag(ui='Checkbox')
    eigen_vectors_ = traitlets.List(traitlets.List(traitlets.CFloat()), help='The eigen vectors corresponding to each feature').tag(output=True)
    eigen_values_ = traitlets.List(traitlets.CFloat(), help='The eigen values that correspond to each feature.').tag(output=True)
    means_ = traitlets.List(traitlets.CFloat(), help='The mean of each feature').tag(output=True)
    explained_variance_ = traitlets.List(traitlets.CFloat(), help='Variance explained by each of the components. Same as the eigen values.').tag(output=True)
    explained_variance_ratio_ = traitlets.List(traitlets.CFloat(), help='Percentage of variance explained by each of the selected components.').tag(output=True)

    def fit(self, df, progress=None):
        '''Fit the PCA model to the DataFrame.

        :param df: A vaex DataFrame.
        :param progress: If True or 'widget', display a progressbar of the fitting process.
        '''
        self.n_components = self.n_components or len(self.features)
        assert self.n_components >= 2, 'At least two features are required.'
        assert self.n_components <= len(self.features), 'Can not have more components than features.'
        C = df.cov(self.features, progress=progress)
        eigen_values, eigen_vectors = np.linalg.eigh(C)
        indices = np.argsort(eigen_values)[::-1]
        self.means_ = df.mean(self.features, progress=progress).tolist()
        self.eigen_vectors_ = eigen_vectors[:, indices].tolist()
        self.eigen_values_ = eigen_values[indices].tolist()
        self.explained_variance_ = self.eigen_values_
        self.explained_variance_ratio_ = (eigen_values[indices] / np.sum(eigen_values)).tolist()

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
            vector = eigen_vectors[:, i]
            if self.whiten:
                expr = copy.func.dot_product(expressions, vector)
                expr = f'({expr}) / {np.sqrt(self.explained_variance_[i])}'
            else:
                expr = copy.func.dot_product(expressions, vector)
            name = self.prefix + str(i + name_prefix_offset)
            copy[name] = expr
        return copy


@register
@generate.register
class PCAIncremental(PCA):
    '''Transform a set of features using the "sklearn.decomposition.IncrementalPCA" algorithm.

    Note that you need to have scikit-learn installed to fit this Transformer, but not
    for transformations using an already fitted Transformer.

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10])
    >>> df
    #    x    y
    0    2   -2
    1    5    3
    2    7    0
    3    2    0
    4   15   10
    >>> pca = vaex.ml.PCAIncremental(n_components=2, features=['x', 'y'], batch_size=3)
    >>> pca.fit_transform(df)
    #    x    y      PCA_0      PCA_1
    0    2   -2  -5.92532   -0.413011
    1    5    3  -0.380494   1.39112
    2    7    0  -0.840049  -2.18502
    3    2    0  -4.61287    1.09612
    4   15   10  11.7587     0.110794
    '''
    snake_name = 'pca_incremental'
    batch_size = traitlets.Int(default_value=1000, help='Number of samples to be send to the transformer in each batch.')
    noise_variance_ = traitlets.CFloat(default_value=0, help='The estimated noise covariance following the Probabilistic PCA model from Tipping and Bishop 1999.').tag(output=True)
    n_samples_seen_ = traitlets.CInt(default_value=0, help='The number of samples processed by the transformer.').tag(output=True)

    def fit(self, df, progress=None):
        '''Fit the PCAIncremental model to the DataFrame.

        :param df: A vaex DataFrame.
        :param progress: If True or 'widget', display a progressbar of the fitting process.
        '''

        self.n_components = self.n_components or len(self.features)

        n_samples = len(df)
        progressbar = vaex.utils.progressbars(progress)
        pca = sklearn.decomposition.IncrementalPCA(n_components=self.n_components,
                                                   batch_size=self.batch_size,
                                                   whiten=self.whiten)

        for i1, i2, chunk in df.evaluate_iterator(self.features, chunk_size=self.batch_size, array_type='numpy'):
            progressbar(i1 / n_samples)
            chunk = np.array(chunk).T.astype(np.float64)
            pca.partial_fit(X=chunk, check_input=False)
        progressbar(1.0)

        self.singular_values_ = pca.singular_values_.tolist()
        self.eigen_vectors_ = pca.components_.T.tolist()
        self.eigen_values_ = pca.explained_variance_.tolist()
        self.explained_variance_ = pca.explained_variance_.tolist()
        self.explained_variance_ratio_ = pca.explained_variance_ratio_.tolist()
        self.means_ = pca.mean_.tolist()
        self.noise_variance_ = pca.noise_variance_
        self.n_samples_seen_ = pca.n_samples_seen_


@register
@generate.register
class RandomProjections(Transformer):
    '''Reduce dimensionality through a random matrix projection.

    The random projections method is based on the Johnson-Lindenstrauss lemma.
    For mode details see https://en.wikipedia.org/wiki/Johnson%E2%80%93Lindenstrauss_lemma

    Note that you need scikit-learn to fit this Transformer but not for transformations using an already fitter Transformer.

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(x=[2,5,7,2,15], y=[-2,3,0,0,10], z=[2, -10, 2, 3, 0])
    >>> df
    #    x    y    z
    0    2   -2    2
    1    5    3  -10
    2    7    0    2
    3    2    0    3
    4   15   10    0
    >>> rand_proj = vaex.ml.RandomProjections(features=['x', 'y', 'z'], n_components=2)
    >>> rand_proj.fit_transform(df)
    #    x    y    z    random_projection_0    random_projection_1
    0    2   -2    2                1.73363             -0.0700273
    1    5    3  -10              -17.8742             -14.0226
    2    7    0    2               -3.32911             -8.50181
    3    2    0    3                2.04843             -1.27538
    4   15   10    0              -17.0289             -28.6562
    '''
    snake_name = 'random_projections'
    n_components = traitlets.CInt(default_value=None, allow_none=True, help='Number of components to retain. If None (default) the number will be set via the Johnson-Lindenstrauss formula. See https://scikit-learn.org/stable/modules/generated/sklearn.random_projection.johnson_lindenstrauss_min_dim.html for more details.')
    eps = traitlets.Float(default_value=0.1, allow_none=True, help='Parameter to control the quality of the embedding according to the Johnson-Lindenstrauss lemma when `n_components` is set to None. The value must be positive.')
    matrix_type = traitlets.Enum(values=['gaussian', 'sparse'], default_value='gaussian', help='The type of random matrix to create. The values can be "gaussian" and "sparse".')
    density = traitlets.Float(default_value=None, allow_none=True, help='Ratio in the range (0, 1] of non-zero component in the random projection matrix. Only valid if `matrix_type` is "sparse". If density is None, the value is set to the minimum density as recommended by Ping Li et al.: 1 / sqrt(n_features).')
    prefix = traitlets.Unicode(default_value="random_projection_", help=help_prefix)
    random_state = traitlets.Int(default_value=None, allow_none=True, help='Controls the pseudo random number generator used to generate the projection matrix at fit time. Used to get reproducible results.')
    random_matrix_ = traitlets.List(traitlets.List(traitlets.CFloat()), help='The random matrix.').tag(output=True)

    @traitlets.validate('eps')
    def _valid_eps(self, proposal):
        if (proposal['value'] > 0) & (proposal['value'] < 1):
            return proposal['value']
        else:
            raise traitlets.TraitError('`eps` must be between 0 and 1.')

    @traitlets.validate('density')
    def _valid_density(self, proposal):
        if (proposal['value'] > 0) & (proposal['value'] <= 1):
            return proposal['value']
        else:
            raise traitlets.TraitError('`density` must be 0 < density <= 1.')

    def fit(self, df):
        '''Fit the RandomProjections to the DataFrame.

        :param df: A vaex DataFrame.
        '''
        n_samples = len(df)
        n_features = len(self.features)

        if self.n_components is None:
            self.n_components = sklearn.random_projection.johnson_lindenstrauss_min_dim(n_samples=n_samples,
                                                                                        eps=self.eps)

        if self.matrix_type == 'gaussian':
            self.random_matrix_ = sklearn.random_projection._gaussian_random_matrix(n_components=self.n_components,
                                                                                    n_features=n_features,
                                                                                    random_state=self.random_state).tolist()

        else:
            density = self.density or 'auto'
            self.random_matrix_ = sklearn.random_projection._sparse_random_matrix(n_components=self.n_components,
                                                                                  n_features=n_features,
                                                                                  density=density,
                                                                                  random_state=self.random_state).toarray().tolist()


    def transform(self, df):
        '''Apply the RandomProjection transformation to the DataFrame.

        :param df: A vaex DataFrame

        :return copy: A shallow copy of the DataFrame that includes the RandomProjection components.
        :rtype: DataFrame
        '''
        copy = df.copy()
        random_matrix = np.array(self.random_matrix_)
        name_prefix_offset = 0
        while self.prefix + str(name_prefix_offset) in copy.get_column_names(virtual=True, strings=True):
            name_prefix_offset += 1

        for component in range(self.n_components):
            vector = random_matrix[component]
            feature_expressions = [copy[feat] for feat in self.features]
            expr = copy.func.dot_product(feature_expressions, vector)
            name = self.prefix + str(component + name_prefix_offset)
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
            labels = vaex.array_types.tolist(df[feature].unique())
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
     #  color®
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
            unique_values = vaex.array_types.tolist(df.unique(expression))

            if None in unique_values:
                unique_values.remove(None)
                unique_values.sort()
                unique_values.insert(0, None)  # This is done in place
            else:
                unique_values.sort()
            uniques.append(unique_values)
        self.uniques_ = uniques

        # detect ability to downcast to uint8

    def transform(self, df):
        '''Transform a DataFrame with a fitted OneHotEncoder.

        :param df: A vaex DataFrame.
        :return: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''
        copy = df.copy()
        downcast_uint8 = np.can_cast(self.one, np.uint8) and np.can_cast(self.zero, np.uint8)
        dtype = 'uint8' if downcast_uint8 else None
        # for each feature, add a virtual column for each unique entry
        for i, feature in enumerate(self.features):
            for j, value in enumerate(self.uniques_[i]):
                str_value = str(value) if value is not None else 'missing'
                column_name = self.prefix + feature + '_' + str_value
                if value is None:
                    copy[column_name] = copy.func.where(copy[feature].ismissing(), self.one, self.zero, dtype=dtype)
                elif isinstance(value, np.float) and np.isnan(value):
                    copy[column_name] = copy.func.where(copy[feature].isnan(), self.one, self.zero, dtype=dtype)
                else:
                    copy[column_name] = copy.func.where(copy[feature] == value, self.one, self.zero, dtype=dtype)
        return copy


@register
@generate.register
class MultiHotEncoder(Transformer):
    '''Encode categorical columns according to a binary multi-hot scheme.

    With Multi-Hot Encoder (sometimes called Binary Encoder), the categorical variables are first
    ordinal encoded, and those encodings are converted to a binary number. Each digit of that binary number
    is a separate column, containing either a "0" or a "1". This is can be considered as an improvement
    over the One-Hot encoder as it guards against generating too many new columns when the cardinality of the
    categorical column is high, while effecively removing the ordinality that an Ordinal Encoder would introduce.

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(color=['red', 'green', 'green', 'blue', 'red'])
    >>> df
    #  color
    0  red
    1  green
    2  green
    3  blue
    4  red
    >>> encoder = vaex.ml.MultiHotEncoder(features=['color'])
    >>> encoder.fit_transform(df)
    #  color      color_0    color_1    color_2
    0  red              0          1          1
    1  green            0          1          0
    2  green            0          1          0
    3  blue             0          0          1
    4  red              0          1          1
    '''

    prefix = traitlets.Unicode(default_value='', help=help_prefix).tag(ui='Text')
    labels_ = traitlets.Dict(default_value={}, allow_none=True, help='The ordinal-encoded labels of each feature.').tag(output=True)

    def fit(self, df):
        '''Fit MultiHotEncoder to the DataFrame.

        :param df: A vaex DataFrame.
        '''
        for feature in self.features:
            # Get unique labels
            labels = vaex.array_types.tolist(df[feature].unique())
            n_labels = len(labels)
            if None in labels:
                labels.remove(None)
                labels.sort()
                labels.insert(0, None)  # This is done in place
            else:
                labels.sort()

            labels_dict = dict(zip(labels, np.arange(1, n_labels+1)))
            self.labels_[feature] = labels_dict

    def _get_n_dims(self, n_labels):
        '''Get the number of dimensions for the multi-hot vector, based on the number of unique labels.'''
        return math.floor(math.log2(n_labels)) + 1 + np.mod(n_labels, 2)

    def transform(self, df):
        '''Transform a DataFrame with a fitted MultiHotEncoder.

        :param df: A vaex DataFrame.
        :return: A shallow copy of the DataFrame that includes the encodings.
        :rtype: DataFrame
        '''
        copy = df.copy()
        for feature in self.features:
            tmp = copy[feature].map(self.labels_[feature], default_value=0)
            n_labels = len(self.labels_[feature])
            n_dims = self._get_n_dims(n_labels=n_labels)
            # i is for the order of the features names,
            # j tracks the order of the labels, as it goes backwards.
            for i, j in enumerate(range(n_dims-1, -1, -1)):
                name = f'{self.prefix}{feature}_{i}'
                copy[name] = (tmp >> j) & 1
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
        for i, feature in enumerate(self.features):
            name = self.prefix+feature
            expression = copy[feature]
            if self.with_mean:
                expression -= self.mean_[i]
            if self.with_std:
                expression /= self.std_[i]
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
    snake_name = 'minmax_scaler'
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

        minmax = []
        for feat in self.features:
            minmax.append(df.minmax(feat, delay=True))

        @vaex.delayed
        def assign(minmax):
            self.fmin_ = [elem[0] for elem in minmax]
            self.fmax_ = [elem[1] for elem in minmax]

        assign(minmax)
        df.execute()

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted MinMaxScaler.

        :param df: A vaex DataFrame.

        :return copy: a shallow copy of the DataFrame that includes the scaled features.
        :rtype: DataFrame
        '''

        copy = df.copy()

        for i, feature in enumerate(self.features):
            name = self.prefix + feature
            a = self.feature_range[0]
            b = self.feature_range[1]
            expr = copy[feature]
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
        for i, feature in enumerate(self.features):
            name = self.prefix + feature
            expr = copy[feature]
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
        for i, feature in enumerate(self.features):
            name = self.prefix+feature
            expr = copy[feature]
            if self.with_centering:
                expr -= self.center_[i]
            if self.with_scaling:
                expr /= self.scale_[i]
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


@register
@generate.register
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


@register
@generate.register
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
            agg = df.groupby(feature, agg={'positive': vaex.agg.mean(self.target)}, sort=True)
            agg['positive'] = agg.func.where(agg['positive'] == 0, self.epsilon, agg['positive'])
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


@register
@generate.register
class KBinsDiscretizer(Transformer):
    '''Bin continous features into discrete bins.

    A stretegy to encode continuous features into discrete bins. The transformed
    columns contain the bin label each sample falls into. In a way this
    transformer Label/Ordinal encodes continous features.

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df = vaex.from_arrays(x=[0, 2.5, 5, 7.5, 10, 12.5, 15])
    >>> bin_trans = vaex.ml.KBinsDiscretizer(features=['x'], n_bins=3, strategy='uniform')
    >>> bin_trans.fit_transform(df)
      #     x    binned_x
      0   0             0
      1   2.5           0
      2   5             1
      3   7.5           1
      4  10             2
      5  12.5           2
      6  15             2
    '''
    snake_name = 'kbins_discretizer'
    n_bins = traitlets.Int(allow_none=False, default_value=5, help='Number of bins. Must be greater than 1.')
    strategy = traitlets.Enum(values=['uniform', 'quantile', 'kmeans'], default_value='uniform', help='Strategy used to define the widths of the bins. Can be either "uniform", "quantile" or "kmeans".')
    prefix = traitlets.Unicode(default_value='binned_', help=help_prefix)
    epsilon = traitlets.Float(default_value=1e-8, allow_none=False, help='Tiny value added to the bin edges ensuring samples close to the bin edges are binned correcly.')
    n_bins_ = traitlets.Dict(help='Number of bins per feature.').tag(output=True)
    bin_edges_ = traitlets.Dict(help='The bin edges for each binned feature').tag(output=True)

    def fit(self, df):
        '''
        Fit KBinsDiscretizer to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        # We need at least two bins to do the transformations
        assert self.n_bins > 1, ' Kwarg `n_bins` must be greated than 1.'

        # Find the extent of the features
        minmax = []
        minmax_promise = []
        for feat in self.features:
            minmax_promise.append(df.minmax(feat, delay=True))

        @vaex.delayed
        def assign(minmax_promise):
            for elem in minmax_promise:
                minmax.append(elem)

        assign(minmax_promise)
        df.execute()

        # warning: everyting is cast to float, which is unavoidable due to the addition of self.epsilon
        minmax = np.array(minmax)
        minmax[:, 1] = minmax[:, 1] + self.epsilon

        # # Determine the bin edges and number of bins depending on the strategy per feature
        if self.strategy == 'uniform':
            bin_edges = {feat: np.linspace(minmax[i, 0], minmax[i, 1], self.n_bins+1) for i, feat in enumerate(self.features)}

        elif self.strategy == 'quantile':
            percentiles = np.linspace(0, 100, self.n_bins + 1)
            bin_edges = df.percentile_approx(self.features, percentage=percentiles)
            bin_edges = {feat: edges for feat, edges in zip(self.features, bin_edges)}

        else:
            from .cluster import KMeans

            bin_edges = {}
            for i, feat in enumerate(self.features):

                # Deterministic initialization with uniform spacing
                uniform_edges = np.linspace(minmax[i, 0], minmax[i, 1], self.n_bins+1)
                centers_init = ((uniform_edges[1:] + uniform_edges[:-1]) * 0.5).tolist()
                centers_init = [[elem] for elem in centers_init]

                # KMeans strategy
                km = KMeans(n_clusters=self.n_bins, init=centers_init, n_init=1, features=[feat])
                km.fit(df)
                # Get and sort the centres of the kmeans clusters
                centers = np.sort(np.array(km.cluster_centers).flatten())
                # Put the bin edges half way between each center (ignoring the outermost edges)
                be = (centers[1:] + centers[:-1]) * 0.5
                # The outermost edges are defined by the min/max of each feature
                # Quickly build a numpy array by concat individual values (min/max) and arrays (be)
                bin_edges[feat] = np.r_[minmax[i, 0], be, minmax[i, 1]]

        # Remove bins whose width are too small (i.e., <= 1e-8)
        n_bins = {}  # number of bins per features that are actually used
        for feat in self.features:
            mask = np.diff(bin_edges[feat], append=np.inf) > 1e-8
            be = bin_edges[feat][mask]
            if len(be) - 1 != self.n_bins:
                warnings.warn(f'Bins whose width are too small (i.e., <= 1e-8) in   {feat} are removed.'
                              f'Consider decreasing the number of bins.')
                bin_edges[feat] = be
            n_bins[feat] = len(be) - 1

        self.bin_edges_ = bin_edges
        self.n_bins_ = n_bins

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted KBinsDiscretizer.

        :param df: A vaex DataFrame.

        :returns copy: a shallow copy of the DataFrame that includes the binned features.
        :rtype: DataFrame
        '''

        df = df.copy()

        for feat in self.features:
            name = self.prefix + feat
            # Samples outside the bin range are added to the closest bin
            df[name] = (df[feat].digitize(self.bin_edges_[feat]) - 1).clip(0, self.n_bins_[feat] - 1)

        return df


@register
@generate.register
class GroupByTransformer(Transformer):
    '''The GroupByTransformer creates aggregations via the groupby operation, which are
    joined to a DataFrame. This is useful for creating aggregate features.

    Example:

    >>> import vaex
    >>> import vaex.ml
    >>> df_train = vaex.from_arrays(x=['dog', 'dog', 'dog', 'cat', 'cat'], y=[2, 3, 4, 10, 20])
    >>> df_test = vaex.from_arrays(x=['dog', 'cat', 'dog', 'mouse'], y=[5, 5, 5, 5])
    >>> group_trans = vaex.ml.GroupByTransformer(by='x', agg={'mean_y': vaex.agg.mean('y')}, rsuffix='_agg')
    >>> group_trans.fit_transform(df_train)
      #  x      y  x_agg      mean_y
      0  dog    2  dog             3
      1  dog    3  dog             3
      2  dog    4  dog             3
      3  cat   10  cat            15
      4  cat   20  cat            15
    >>> group_trans.transform(df_test)
      #  x        y  x_agg    mean_y
      0  dog      5  dog      3.0
      1  cat      5  cat      15.0
      2  dog      5  dog      3.0
      3  mouse    5  --       --
    '''

    snake_name = 'groupby_transformer'
    by = traitlets.Unicode(allow_none=False, help='The feature on which to do the grouping.')
    agg = traitlets.Dict(help='Dict where the keys are feature names and the values are vaex.agg objects.')
    rprefix = traitlets.Unicode(default_value='', help='Prefix for the names of the aggregate features in case of a collision.')
    rsuffix = traitlets.Unicode(default_value='', help='Suffix for the names of the aggregate features in case of a collision.')
    df_group_ = traitlets.Instance(klass=vaex.dataframe.DataFrame, allow_none=True)

    def fit(self, df):
        '''
        Fit GroupByTransformer to the DataFrame.

        :param df: A vaex DataFrame.
        '''

        if not self.agg:
            raise ValueError('You have to specify a dict for the `agg` keyword.')
        if len(self.by)==0:
            raise ValueError('Please specify a value for the `by` keyword.')
        self.df_group_ = df.groupby(by=self.by, agg=self.agg)

    def transform(self, df):
        '''
        Transform a DataFrame with a fitted GroupByTransformer.

        :param df: A vaex DataFrame.

        :returns copy: a shallow copy of the DataFrame that includes the aggregated features.
        :rtype: DataFrame
        '''

        df = df.copy()
        # We effectively want to do a join, but since that is not part of the state, it will not be state
        # transferrable, instead we implement this with map
        # df = df.join(other=self.df_group_, on=self.by, how='left', rprefix=self.rprefix, rsuffix=self.rsuffix)
        key_values = self.df_group_[self.by].tolist()
        for name in self.df_group_.get_column_names():
            if name == self.by:
                continue  # we don't need to include the column we group/join on
            mapper = dict(zip(key_values, self.df_group_[name].values))
            join_name = name
            if join_name in df:
                join_name = self.rprefix + join_name + self.rsuffix
            df[join_name] = df[self.by].map(mapper, allow_missing=True)
        return df

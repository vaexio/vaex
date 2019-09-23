import numpy as np
import traitlets
import vaex.ml.state
import logging
import vaex
import vaex.serialize
from numba import jit
from . import generate
# vaex.set_log_level_debug()

logger_km = logging.getLogger('vaex.ml.kmeans')


def Matrix(type=traitlets.CFloat):
    return traitlets.List(traitlets.List(type()))


# @jit('void(double[:], double[:], double[:])', nopython=True, nogil=True)
@jit(nopython=True, nogil=True, cache=True)
def distances_square(result, centroids, *blocks):
    """
    Function under test.
    """
    k = centroids.shape[0]
    dimensions = centroids.shape[1]
    N = result.shape[1]
    for i in range(k):
        for d in range(dimensions):
            for j in range(N):
                result[i, j] += (blocks[d][j] - centroids[i][d])**2


@jit(nopython=True, nogil=True, cache=True)
def centroid_stats(centroids, counts, sumpos, inertia, *blocks):
    # classes = np.argmin(distances_sq, axis=0)
    # distances_sq = np.choose(classes, distances_sq)
    # inertia = (distances_sq).sum()
    # inertia = 0
    N = blocks[0].shape[0]
    runs = sumpos.shape[0]
    clusters = sumpos.shape[1]
    dimensions = sumpos.shape[2]
    for j in range(N):
        for run in range(runs):
            # if done[run]: ## TODO: bug in numba? seems to crash if included
                best_distance = 1e100
                best_class = 10000
                for cluster in range(clusters):
                    distance = 0
                    for d in range(dimensions):
                        distance += (centroids[run, cluster, d] - blocks[d][j])**2
                    if distance < best_distance:
                        best_class = cluster
                        best_distance = distance
                cls = best_class  # classes[j]
                inertia[run] += best_distance
                for d in range(dimensions):
                    sumpos[run, cls, d] += blocks[d][j]
                counts[run, cls] += 1
#    return inertia
        # for i in range(k):


@vaex.serialize.register
@generate.register
class KMeans(vaex.ml.state.HasState):
    '''The KMeans clustering algorithm.

    Example:

    >>> import vaex.ml
    >>> import vaex.ml.cluster
    >>> df = vaex.ml.datasets.load_iris()
    >>> features = ['sepal_width', 'petal_length', 'sepal_length', 'petal_width']
    >>> cls = vaex.ml.cluster.KMeans(n_clusters=3, features=features, init='random', max_iter=10)
    >>> cls.fit(df)
    >>> df = cls.transform(df)
    >>> df.head(5)
     #    sepal_width    petal_length    sepal_length    petal_width    class_    prediction_kmeans
     0            3               4.2             5.9            1.5         1                    2
     1            3               4.6             6.1            1.4         1                    2
     2            2.9             4.6             6.6            1.3         1                    2
     3            3.3             5.7             6.7            2.1         2                    0
     4            4.2             1.4             5.5            0.2         0                    1
    '''
    features = traitlets.List(traitlets.Unicode(), help='List of features to cluster.')
    n_clusters = traitlets.CInt(default_value=2, help='Number of clusters to form.')
    init = traitlets.Union([Matrix(), traitlets.Unicode()], default_value='random', help='Method for initializing the centroids.')
    n_init = traitlets.CInt(default_value=1, help='Number of centroid initializations. \
                                                   The KMeans algorithm will be run for each initialization, \
                                                   and the final results will be the best output of the n_init \
                                                   consecutive runs in terms of inertia.')
    max_iter = traitlets.CInt(default_value=300, help='Maximum number of iterations of the KMeans algorithm for a single run.')
    random_state = traitlets.CInt(default_value=None, allow_none=True, help='Random number generation for centroid initialization. \
                                                                             If an int is specified, the randomness becomes deterministic.')
    verbose = traitlets.CBool(default_value=False, help='If True, enable verbosity mode.')
    cluster_centers = traitlets.List(traitlets.List(traitlets.CFloat()), help='Coordinates of cluster centers.')
    inertia = traitlets.CFloat(default_value=None, allow_none=True, help='Sum of squared distances of samples to their closest cluster center.')
    prediction_label = traitlets.Unicode(default_value='prediction_kmeans', help='The name of the virtual column that houses the cluster labels for each point.')

    def __call__(self, *blocks):
        return self._calculate_classes(*blocks)

    def _calculate_distances_squared(self, *blocks):
        N = len(blocks[0])  # they are all the same length
        centroids = np.array(self.cluster_centers)
        k = centroids.shape[0]
        dimensions = centroids.shape[1]
        distances_sq = np.zeros((k, N))
        if 1:
            distances_square(distances_sq, centroids, *blocks)
        else:
            for d in range(dimensions):
                for i in range(k):
                    distances_sq[i] += (blocks[d] - centroids[i][d])**2
        return distances_sq

    def _calculate_classes(self, *blocks):
        distances_sq = self._calculate_distances_squared(*blocks)
        classes = np.argmin(distances_sq, axis=0)
        return classes

    def generate_cluster_centers_random(self, dataframe, rng):
        indices = rng.randint(0, len(dataframe), self.n_clusters)
        return [[dataframe.evaluate(feature, i1=i, i2=i+1)[0] for feature in self.features] for i in indices]

    def transform(self, dataframe):
        '''
        Label a DataFrame with a fitted KMeans model.

        :param dataframe: A vaex DataFrame.

        :returns copy: A shallow copy of the DataFrame that includes the cluster labels.
        :rtype: DataFrame
        '''
        copy = dataframe.copy()
        lazy_function = copy.add_function('kmean_predict_function', self, unique=True)
        expression = lazy_function(*self.features)
        copy.add_virtual_column(self.prediction_label, expression, unique=False)
        return copy

    def fit(self, dataframe):
        '''
        Fit the KMeans model to the dataframe.

        :param dataframe: A vaex DataFrame.
        '''
        if self.init == 'random':
            rng = np.random.RandomState(self.random_state)
            self.run_cluster_centers = [self.generate_cluster_centers_random(dataframe, rng) for k in range(self.n_init)]
        else:
            if self.n_init > 1:
                print("WARNING: n_init > 1 , but init given, only doing one run")
            self.run_cluster_centers = [self.init]
        done = [False] * len(self.run_cluster_centers)
        alldone = False
        first = True
        previous_inertias = None
        iteration = 0
        inertias_list = []
        while not alldone:
            new_centers, inertias = self._find_centers_and_inertias(dataframe, done)
            if self.verbose:
                inertia_msges = [('--' if d else '{: 3}'.format(k)) for k, d in zip(inertias, done)]
                if len(inertias) == 1:
                    inertia_msg = inertia_msges[0]
                else:
                    inertia_msg = " | ".join(inertia_msges)
                print('Iteration {: 4}, inertia {}'.format(iteration, inertia_msg))
            if not first:  # we can only to a check after the second iteration
                done = self._is_done(previous_inertias, inertias)
                alldone = np.all(done)
            else:
                first = False
            iteration += 1
            if (iteration >= self.max_iter):
                logger_km.debug('reached max iterations: %s', self.max_iter)
                alldone = True
            previous_inertias = inertias
            inertias_list.append(inertias)
            self.run_cluster_centers = new_centers
        best_run = np.argmin(inertias)
        self.inertia = inertias[best_run]
        self.inertias = np.array(inertias_list)[:, best_run]
        self.cluster_centers = new_centers[best_run]

    def _is_done(self, inertias1, inertias2):
        diffs = [(inertia1 - inertia2) for inertia1, inertia2 in zip(inertias1, inertias2)]
        return [(diff < 1e-3) for diff in diffs]

    def _find_centers_and_inertias(self, dataframe, done):
        done = np.array(done, dtype=np.int8)
        centroids = np.array(self.run_cluster_centers)
        runs = centroids.shape[0]
        clusters = centroids.shape[1]
        dimensions = centroids.shape[2]
        # print("k =", k)
        assert dimensions == len(self.features), "nr of dimensions for centroid should equal nr of features"

        def map(*blocks):  # this will be called with a chunk of the data
            sumpos = np.zeros((runs, clusters, dimensions))
            counts = np.zeros((runs, clusters))
            inertia = np.zeros((runs))
            if 1:
                centroid_stats(centroids, counts, sumpos, inertia,  *blocks)
            else:
                # this is the pure python code
                # although not made for multiple runs yet
                distances_sq = self._calculate_distances_squared(*blocks)
                classes = np.argmin(distances_sq, axis=0)
                distances_sq = np.choose(classes, distances_sq)
                inertia = (distances_sq).sum()
                for i in range(k):
                    mask = classes == i
                    counts[i] = np.sum(mask)
                    for d in range(dimensions):
                        sumpos[i, d] += blocks[d][mask].sum()
            return sumpos, counts, inertia

        def reduce(x, y):
            sumpos1, counts1, inertia1 = x
            sumpos2, counts2, inertia2 = y
            return sumpos1 + sumpos2, counts1 + counts2, inertia1 + inertia2
        sumpos, counts, inertia = dataframe.map_reduce(map, reduce, self.features)
        means = (sumpos/counts[:, :, np.newaxis])
        return means.tolist(), inertia

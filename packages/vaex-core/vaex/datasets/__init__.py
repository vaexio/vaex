import vaex.utils
import vaex
import os
import numpy as np
# data_dir = "/tmp/vaex/data"
data_dir = vaex.settings.main.data.path

try:
    from urllib import urlretrieve  # py2
except ImportError:
    from urllib.request import urlretrieve  # py3


def _url_to_filename(url, replace_ext=None, subdir=None):
    if subdir:
        filename = os.path.join(data_dir, subdir, url.split("/")[-1])
    else:
        filename = os.path.join(data_dir, url.split("/")[-1])
    if replace_ext:
        dot_index = filename.rfind(".")
        filename = filename[:dot_index] + replace_ext
    return filename


class Hdf5Download(object):
    def __init__(self, url):
        self.url = url
        self.url_list = [self.url]

    @property
    def filename(self):
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, _url_to_filename(self.url))

    def download(self, force=False):
        if not os.path.exists(self.filename) or force:
            print("Downloading %s to %s" % (self.url, self.filename))
            code = os.system(self.wget_command(0))
            if not os.path.exists(self.filename):
                print("wget failed, using urlretrieve")
                self.download_urlretrieve()

    def download_urlretrieve(self):
        urlretrieve(self.url, self.filename)

    def fetch(self, force_download=False):
        self.download(force=force_download)
        return vaex.open(self.filename, convert=not self.filename.endswith('.hdf5'), progress=True)

    def wget_command(self, i):
        assert i == 0
        url = self.url_list[i]
        return "wget --progress=bar:force -c -P %s %s" % (data_dir, url)


def iris():
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

    >>> import vaex.datasets
    >>> df = vaex.datasets.datasets.iris()
    >>> df.describe()
    '''
    dirname = os.path.dirname(__file__)
    return vaex.open(os.path.join(dirname, 'iris.hdf5'))


def _iris(name, iris_previous, N):
    filename = os.path.join(vaex.settings.data.path, name + '.hdf5')
    if os.path.exists(filename):
        return vaex.open(filename)
    else:
        iris = iris_previous()
        repeat = int(np.ceil(N / len(iris)))
        ds = vaex.concat([iris] * repeat)
        ds.export_hdf5(filename)
        return vaex.open(filename)


def iris_subsample(N, error_percentage=5, ds=None):
    '''Returns the iris set repeated so it include ~1e4 rows'''
    # return _iris_subsample('iris_1e4', iris, int(1e4))
    ds = ds or iris()
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


def iris_1e4():
    '''Returns the iris set repeated so it include ~1e4 rows'''
    return _iris('iris_1e4', iris, int(1e4))


def iris_1e5():
    '''Returns the iris set repeated so it include ~1e5 rows'''
    return _iris('iris_1e5', iris_1e4, int(1e5))


def iris_1e6():
    '''Returns the iris set repeated so it include ~1e6 rows'''
    return _iris('iris_1e6', iris_1e5, int(1e6))


def iris_1e7():
    '''Returns the iris set repeated so it include ~1e7 rows'''
    return _iris('iris_1e7', iris_1e6, int(1e7))


def iris_1e8():
    '''Returns the iris set repeated so it include ~1e8 rows'''
    return _iris('iris_1e8', iris_1e7, int(1e8))


def iris_1e9():
    '''Returns the iris set repeated so it include ~1e8 rows'''
    return _iris('iris_1e9', iris_1e8, int(1e9))


def titanic():
    '''
    Returns the classic Titanic dataset.

    Description of the columns can be found in dataset.description.

    Example:

    >>> import vaex.datasets
    >>> df = vaex.datasets.titanic()
    >>> print(df.description)
    >>> df.describe()
    '''
    dirname = os.path.dirname(__file__)
    return vaex.open(os.path.join(dirname, 'titanic.hdf5'))


def taxi():
    '''One year of NYC Yellow Cab data.

    The original raw data can be downloaded from https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
    '''
    return Hdf5Download("https://github.com/vaexio/vaex-datasets/releases/download/1.1/yellow_taxi_2012_zones.parquet").fetch()


def helmi_simulation_data(full=False):
    '''Result of an N-body simulation of the accretion of 33 satellite galaxies into a Milky Way dark matter halo.

    Data was greated by Helmi & de Zeeuw 2000.
    The data contains the position (x, y, z), velocitie (vx, vy, vz), the energy (E),
    the angular momentum (L, Lz) and iron content (FeH) of the particles.

    :param bool full: If True, it returns the full output of the simulation, i.e 3.3 million rows. Otherwise only 10% is returned.
    :rtype: DataFrame
    '''
    if full is False:
        return Hdf5Download("https://github.com/vaexio/vaex-datasets/releases/download/v1.0/helmi-dezeeuw-2000-FeH-v2-10percent.hdf5").fetch()
    return Hdf5Download("https://github.com/vaexio/vaex-datasets/releases/download/v1.0/helmi-dezeeuw-2000-FeH-v2.hdf5").fetch()


def tgas(full=False):
    '''The Tycho-Gaia Astronomical Solution Dataset.

    This dataset combines data of the Hipparcos and Tycho catalogues with the Gaia DR1
    catalogue in order to provide a full astrometric solution for the Gaia data for
    the stars in common.

    :param bool full: If True, it returns the full output of the simulation, i.e 2 million rows. Otherwise only 1% is returned.

    For more information about this datasets visit https://gea.esac.esa.int/archive/documentation/GDR1/Data_processing/chap_cu3tyc/
    '''
    if full:
        return Hdf5Download("https://github.com/vaexio/vaex-datasets/releases/download/v1.0/tgas.hdf5").fetch()
    else:
        return Hdf5Download("https://github.com/vaexio/vaex-datasets/releases/download/v1.0/tgas_1percent.hdf5").fetch()


# TODO: deprecate in v5
helmi_de_zeeuw = Hdf5Download("https://github.com/vaexio/vaex-datasets/releases/download/v1.0/helmi-dezeeuw-2000-FeH-v2.hdf5")

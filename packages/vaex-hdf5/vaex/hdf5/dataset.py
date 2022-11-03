__author__ = 'maartenbreddels'
import os
import logging
import pyarrow as pa
import re

import numpy as np
import numpy.ma

import vaex
from vaex.utils import ensure_string
import vaex.dataset
import vaex.file
from vaex.dataset_mmap import DatasetMemoryMapped
import vaex.column
from vaex.column import ColumnNumpyLike, ColumnStringArrow
import vaex.arrow.convert
import vaex.array_types

astropy = vaex.utils.optional_import("astropy.units")


logger = logging.getLogger("vaex.file")

dataset_type_map = {}

# h5py doesn't want to build at readthedocs
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
try:
    import h5py
except:
    if not on_rtd:
        raise


def _try_unit(unit):
    try:
        unit = astropy.units.Unit(str(unit))
        if not isinstance(unit, astropy.units.UnrecognizedUnit):
            return unit
    except:
        # logger.exception("could not parse unit: %r", unit)
        pass
    try:
        unit_mangle = re.match(".*\[(.*)\]", str(unit)).groups()[0]
        unit = astropy.units.Unit(unit_mangle)
    except:
        pass  # logger.exception("could not parse unit: %r", unit)
    if isinstance(unit, str):
        return None
    elif isinstance(unit, astropy.units.UnrecognizedUnit):
        return None
    else:
        return unit

@vaex.dataset.register
class Hdf5MemoryMapped(DatasetMemoryMapped):
    snake_name = "hdf5"
    """Implements the vaex hdf5 file format"""

    def __init__(self, path, write=False, fs_options={}, fs=None, nommap=None, group=None, _fingerprint=None):
        if nommap is None:
            nommap = (not vaex.file.memory_mappable(path)) or fs is not None
        super(Hdf5MemoryMapped, self).__init__(vaex.file.stringyfy(path), write=write, nommap=nommap, fs_options=fs_options, fs=fs)
        self._cached_fingerprint = _fingerprint  # used during test, where we don't want to trigger an s3 call
        self._all_mmapped = True
        self._open(path)
        self.units = {}
        self.group = group
        self._version = 1
        self._load()
        if not write:  # in write mode, call freeze yourself, so the hashes are computed
            self._freeze()
        else:
            # make sure we set the row count, which otherwise freeze would do
            self._set_row_count()
        if self._all_mmapped:
            self.h5file.close()

    def _open(self, path):
        if vaex.file.is_file_object(path):
            file = path  # support file handle for testing
            self.file_map[self.path] = file
        else:
            mode = 'rb+' if self.write else 'rb'
            file = vaex.file.open(self.path, mode=mode, fs_options=self.fs_options, fs=self.fs, for_arrow=False)
            self.file_map[self.path] = file
        self.h5file = h5py.File(file, "r+" if self.write else "r")

    def __getstate__(self):
        state = {'group': self.group, **super().__getstate__()}
        return state

    def __setstate__(self, state):
        self.group = state.pop('group')
        super().__setstate__(state)
        self._open(self.path)
        self._load()
        self._freeze()

    def write_meta(self):
        """ucds, descriptions and units are written as attributes in the hdf5 file, instead of a seperate file as
         the default :func:`Dataset.write_meta`.
         """
        with h5py.File(self.path, "r+") as h5file_output:
            h5table_root = h5file_output[self.group]
            if self.description is not None:
                h5table_root.attrs["description"] = self.description
            h5columns = h5table_root if self._version == 1 else h5table_root['columns']
            for column_name in self.columns.keys():
                h5dataset = None
                if column_name in h5columns:
                    h5dataset = h5columns[column_name]
                else:
                    for group in h5columns.values():
                        if 'type' in group.attrs:
                            if group.attrs['type'] in ['csr_matrix']:
                                for name, column in group.items():
                                    if name == column_name:
                                        h5dataset = column
                if h5dataset is None:
                    raise ValueError('column {} not found'.format(column_name))

                aliases = [key for key, value in self._column_aliases.items() if value == column_name]
                aliases.sort()
                if aliases:
                    h5dataset.attrs['alias'] = aliases[0]
                for name, values in [("ucd", self.ucds), ("unit", self.units), ("description", self.descriptions)]:
                    if column_name in values:
                        value = ensure_string(values[column_name], cast=True)
                        h5dataset.attrs[name] = value
                    else:
                        if name in h5columns.attrs:
                            del h5dataset.attrs[name]

    @classmethod
    def create(cls, path, N, column_names, dtypes=None, write=True):
        """Create a new (empty) hdf5 file with columns given by column names, of length N

        Optionally, numpy dtypes can be passed, default is floats
        """

        dtypes = dtypes or [np.float] * len(column_names)

        if N == 0:
            raise ValueError("Cannot export empty table")
        with h5py.File(path, "w") as h5file_output:
            h5data_output = h5file_output.require_group("data")
            for column_name, dtype in zip(column_names, dtypes):
                shape = (N,)
                print(dtype)
                if dtype.type == np.datetime64:
                    array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=np.int64, track_times=False)
                    array.attrs["dtype"] = dtype.name
                else:
                    array = h5file_output.require_dataset("/data/%s" % column_name, shape=shape, dtype=dtype, track_times=False)
                array[0] = array[0]  # make sure the array really exists
        return Hdf5MemoryMapped(path, write=write)

    @classmethod
    def quick_test(cls, path, fs_options={}, fs=None):
        path, options = vaex.file.split_options(path)
        return path.endswith('.hdf5') or path.endswith('.h5')

    @classmethod
    def can_open(cls, path, fs_options={}, fs=None, group=None, **kwargs):
        with vaex.file.open(path, fs_options=fs_options, fs=fs) as f:
            signature = f.read(4)
            if signature != b"\x89\x48\x44\x46":
                return False
            with h5py.File(f, "r") as h5file:
                root_datasets = [dataset for name, dataset in h5file.items() if isinstance(dataset, h5py.Dataset)]
                return ("data" in h5file) or ("columns" in h5file) or ("table" in h5file) or \
                       (group is not None and group in h5file) or \
                       (len(root_datasets) > 0)
        return False

    @classmethod
    def get_options(cls, path):
        return []

    @classmethod
    def option_to_args(cls, option):
        return []

    def _load(self):
        self.ucds = {}
        self.descriptions = {}
        self.units = {}

        if self.group is None:
            if "data" in self.h5file:
                self._load_columns(self.h5file["/data"])
                self.group = "/data"
            if "table" in self.h5file:
                self._version = 2
                self._load_columns(self.h5file["/table"])
                self.group = "/table"
            root_datasets = [dataset for name, dataset in self.h5file.items() if isinstance(dataset, h5py.Dataset)]
            if len(root_datasets):
                # if we have datasets at the root, we assume 'version 1'
                self._load_columns(self.h5file)
                self.group = "/"

            # TODO: shall we rename it vaex... ?
            # if "vaex" in self.h5file:
            # self.load_columns(self.h5file["/vaex"])
            # h5table_root = "/vaex"
            if "columns" in self.h5file:
                self._load_columns(self.h5file["/columns"])
                self.group = "/columns"
        else:
            self._version = 2
            self._load_columns(self.h5file[self.group])

        if "properties" in self.h5file:
            self._load_variables(self.h5file["/properties"])  # old name, kept for portability
        if "variables" in self.h5file:
            self._load_variables(self.h5file["/variables"])
        # self.update_meta()
        # self.update_virtual_meta()

    # def
    def _load_axes(self, axes_data):
        for name in axes_data:
            axis = axes_data[name]
            logger.debug("loading axis %r" % name)
            offset = axis.id.get_offset()
            shape = axis.shape
            assert len(shape) == 1  # ony 1d axes
            # print name, offset, len(axis), axis.dtype
            self.addAxis(name, offset=offset, length=len(axis), dtype=axis.dtype)
            # self.axis_names.append(axes_data)
            # self.axes[name] = np.array(axes_data[name])

    def _load_variables(self, h5variables):
        for key, value in list(h5variables.attrs.items()):
            self.variables[key] = value


    def _map_hdf5_array(self, data, mask=None, as_arrow=False):
        offset = data.id.get_offset()
        if len(data) == 0 and offset is None:
            offset = 0 # we don't care about the offset for empty arrays
        if offset is None:  # non contiguous array, chunked arrays etc
            # we don't support masked in this case
            assert as_arrow is False
            column = ColumnNumpyLike(data)
            self._all_mmapped = False
            return column
        else:
            shape = data.shape
            dtype = data.dtype
            if "dtype" in data.attrs:
                # ignore the special str type, which is not a numpy dtype
                if data.attrs["dtype"] != "str":
                    dtype = data.attrs["dtype"]
                    if dtype == 'utf32':
                        dtype = np.dtype('U' + str(data.attrs['dlength']))
            #self.addColumn(column_name, offset, len(data), dtype=dtype)
            array = self._map_array(offset, dtype=dtype, shape=shape)
            if as_arrow:
                if isinstance(array, np.ndarray):
                    array = vaex.array_types.to_arrow(array)
                else:
                    array = vaex.column.ColumnArrowLazyCast(array, vaex.dtype_of(array).arrow)
            if mask is not None:
                if as_arrow:
                    buffers = array.buffers()
                    null_bitmap = pa.py_buffer(self._map_hdf5_array(mask))
                    ar = pa.Array.from_buffers(array.type, len(array), [null_bitmap] + buffers[1:])
                else:
                    mask_array = self._map_hdf5_array(mask)
                    ar = vaex.column.ColumnMaskedNumpy(array, mask_array)
                    if isinstance(array, np.ndarray):
                        ar = np.ma.array(array, mask=mask_array, shrink=False)
                    else:
                        ar = vaex.column.ColumnMaskedNumpy(array, mask_array)
                return ar
            else:
                return array

    def _load_columns(self, h5data, first=[]):
        # print h5data
        # make sure x y x etc are first

        finished = set()
        if "description" in h5data.attrs:
            self.description = ensure_string(h5data.attrs["description"])
        # hdf5, or h5py doesn't keep the order of columns, so manually track that, also enables reordering later
        h5columns = h5data if self._version == 1 else h5data['columns']
        if "column_order" in h5columns.attrs:
            column_order = ensure_string(h5columns.attrs["column_order"]).split(",")
        else:
            column_order = []
        # for name in list(h5columns):
        #     if name not in column_order:
        #         column_order.append(name)
        # for column_name in column_order:
            # if column_name in h5columns and column_name not in finished:
        for group_name in list(h5columns):
            logger.debug('loading column: %s', group_name)
            group = h5columns[group_name]
            if 'type' in group.attrs:
                type = group.attrs['type']
                if type in ['csr_matrix']:
                    from scipy.sparse import csc_matrix, csr_matrix
                    class csr_matrix_nocheck(csr_matrix):
                        def check_format(self, *args, **kwargs):
                            pass
                    data = self._map_hdf5_array(group['data'])
                    indptr = self._map_hdf5_array(group['indptr'])
                    indices = self._map_hdf5_array(group['indices'])
                    #column_names = ensure_string(group.attrs["column_names"]).split(",")
                    # make sure we keep the original order
                    groups = [(name, value) for name, value in group.items() if isinstance(value, h5py.Group)]
                    column_names = [None] * len(groups)
                    for name, column in groups:
                        column_names[column.attrs['column_index']] = name
                    matrix = csr_matrix_nocheck((data, indices, indptr), shape=(len(indptr)-1, len(column_names)))
                    assert matrix.data is data
                    # assert matrix.indptr is indptr
                    assert matrix.indices is indices
                    self.add_columns(column_names, matrix)
                if type == 'dictionary_encoded':
                    index = self._map_column(group['indices'], as_arrow=True)
                    values = self._map_column(group['dictionary'], as_arrow=True)
                    if 'null_bitmap' in group['indices'] or 'mask' in group['indices']:
                        raise ValueError(f'Did not expect null data in encoded column {group_name}')
                    if isinstance(values, vaex.column.Column):
                        encoded = vaex.column.ColumnArrowDictionaryEncoded(index, values)
                    else:
                        encoded = pa.DictionaryArray.from_arrays(index, values, safe=False)
                    self.add_column(group_name, encoded)
                else:
                    raise TypeError(f'Unexpected type {type!r} in {group_name}')
            else:
                column_name = group_name
                column = h5columns[column_name]
                if "alias" in column.attrs:
                    column_name = column.attrs["alias"]
                if "ucd" in column.attrs:
                    self.ucds[column_name] = ensure_string(column.attrs["ucd"])
                if "description" in column.attrs:
                    self.descriptions[column_name] = ensure_string(column.attrs["description"])
                if "unit" in column.attrs:
                    try:
                        unitname = ensure_string(column.attrs["unit"])
                        if unitname and unitname != "None":
                            self.units[column_name] = _try_unit(unitname)
                    except:
                        logger.exception("error parsing unit: %s", column.attrs["unit"])
                if "units" in column.attrs:  # Amuse case
                    unitname = ensure_string(column.attrs["units"])
                    logger.debug("amuse unit: %s", unitname)
                    if unitname == "(0.01 * system.get('S.I.').base('length'))":
                        self.units[column_name] = astropy.units.Unit("cm")
                    if unitname == "((0.01 * system.get('S.I.').base('length')) * (system.get('S.I.').base('time')**-1))":
                        self.units[column_name] = astropy.units.Unit("cm/s")
                    if unitname == "(0.001 * system.get('S.I.').base('mass'))":
                        self.units[column_name] = astropy.units.Unit("gram")

                    if unitname == "system.get('S.I.').base('length')":
                        self.units[column_name] = astropy.units.Unit("m")
                    if unitname == "(system.get('S.I.').base('length') * (system.get('S.I.').base('time')**-1))":
                        self.units[column_name] = astropy.units.Unit("m/s")
                    if unitname == "system.get('S.I.').base('mass')":
                        self.units[column_name] = astropy.units.Unit("kg")
                if self._version == 1:
                    column = self._map_hdf5_array(column)
                    self.add_column(column_name, column)
                elif hasattr(column["data"], "dtype"):
                    column = self._map_column(column)
                    self.add_column(column_name, column)
                    dtype = vaex.dtype_of(column)
                    logger.debug("adding column %r with dtype %r", column_name, dtype)
                else:
                    raise TypeError(f'{group_name} is missing dtype')

        all_columns = dict(**self._columns)
        # in case the column_order refers to non-existing columns
        column_order = [k for k in column_order if k in all_columns]
        column_names = []
        self._columns = {}
        for name in column_order:
            self._columns[name] = all_columns.pop(name)
        # add the rest
        for name, col in all_columns.items():
            self._columns[name] = col
            # self.column_names.append(name)

    def _map_column(self, column : h5py.Group, as_arrow=False):
        data = column["data"]
        if "dtype" in data.attrs and data.attrs["dtype"] == "str":
            indices = self._map_hdf5_array(column['indices'])
            bytes = self._map_hdf5_array(data)
            if "null_bitmap" in column:
                null_bitmap = self._map_hdf5_array(column['null_bitmap'])
            else:
                null_bitmap = None
            if isinstance(indices, np.ndarray):  # this is a real mmappable file
                return vaex.arrow.convert.arrow_string_array_from_buffers(bytes, indices, null_bitmap)
            else:
                # if not a reall mmappable array, we fall back to this, maybe we can generalize this
                return ColumnStringArrow(indices, bytes, null_bitmap=null_bitmap)
        else:
            if self._version > 1 and 'mask' in column:
                return self._map_hdf5_array(data, column['mask'], as_arrow=as_arrow)
            else:
                if 'null_bitmap' in column:
                    return self._map_hdf5_array(data, column['null_bitmap'], as_arrow=True)
                else:
                    return self._map_hdf5_array(data, as_arrow=as_arrow)


    def close(self):
        super().close()
        self.h5file.close()

    def __expose_array(self, hdf5path, column_name):
        array = self.h5file[hdf5path]
        array[0] = array[0]  # without this, get_offset returns None, probably the array isn't really created
        offset = array.id.get_offset()
        self.remap()
        self.addColumn(column_name, offset, len(array), dtype=array.dtype)

    def __add_column(self, column_name, dtype=np.float64, length=None):
        array = self.h5data.create_dataset(column_name, shape=(self._length if length is None else length,), dtype=dtype)
        array[0] = array[0]  # see above
        offset = array.id.get_offset()
        self.h5file.flush()
        self.remap()
        self.addColumn(column_name, offset, len(array), dtype=array.dtype)


dataset_type_map["h5vaex"] = Hdf5MemoryMapped


class AmuseHdf5MemoryMapped(Hdf5MemoryMapped):
    """Implements reading Amuse hdf5 files `amusecode.org <http://amusecode.org/>`_"""

    def __init__(self, path, write=False, fs_options={}, fs=None):
        super(AmuseHdf5MemoryMapped, self).__init__(path, write=write, fs_options=fs_options, fs=fs)

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        h5file = None
        try:
            h5file = h5py.File(path, "r")
        except:
            return False
        if h5file is not None:
            with h5file:
                return ("particles" in h5file)  # or ("columns" in h5file)
        return False

    def _load(self):
        particles = self.h5file["/particles"]
        for group_name in particles:
            # import pdb
            # pdb.set_trace()
            group = particles[group_name]
            self._load_columns(group["attributes"])

            column_name = "keys"
            column = group[column_name]
            offset = column.id.get_offset()
            data = self._map_hdf5_array(column)
            self.add_column(column_name, data)
        # self.update_meta()
        # self.update_virtual_meta()


dataset_type_map["amuse"] = AmuseHdf5MemoryMapped


gadget_particle_names = "gas halo disk bulge stars dm".split()


class Hdf5MemoryMappedGadget(DatasetMemoryMapped):
    """Implements reading `Gadget2 <http://wwwmpa.mpa-garching.mpg.de/gadget/>`_ hdf5 files """

    def __init__(self, path, particle_name=None, particle_type=None, fs_options={}, fs=None):
        if "#" in path:
            path, index = path.split("#")
            index = int(index)
            particle_type = index
            particle_name = gadget_particle_names[particle_type]
        elif particle_type is not None:
            self.particle_name = gadget_particle_names[self.particle_type]
            self.particle_type = particle_type
        elif particle_name is not None:
            if particle_name.lower() in gadget_particle_names:
                self.particle_type = gadget_particle_names.index(particle_name.lower())
                self.particle_name = particle_name.lower()
            else:
                raise ValueError("particle name not supported: %r, expected one of %r" % (particle_name, " ".join(gadget_particle_names)))
        else:
            raise Exception("expected particle type or name as argument, or #<nr> behind path")
        super(Hdf5MemoryMappedGadget, self).__init__(path, fs_options=fs_options, fs=fs)
        self.particle_type = particle_type
        self.particle_name = particle_name
        self.name = self.name + "-" + self.particle_name
        h5file = h5py.File(self.path, 'r')
        # for i in range(1,4):
        key = "/PartType%d" % self.particle_type
        if key not in h5file:
            raise KeyError("%s does not exist" % key)
        particles = h5file[key]
        for name in list(particles.keys()):
            # name = "/PartType%d/Coordinates" % i
            data = particles[name]
            if isinstance(data, h5py.highlevel.Dataset):  # array.shape
                array = data
                shape = array.shape
                if len(shape) == 1:
                    offset = array.id.get_offset()
                    if offset is not None:
                        self.addColumn(name, offset, data.shape[0], dtype=data.dtype)
                else:
                    if name == "Coordinates":
                        offset = data.id.get_offset()
                        if offset is None:
                            print((name, "is not of continuous layout?"))
                            sys.exit(0)
                        bytesize = data.dtype.itemsize
                        self.addColumn("x", offset, data.shape[0], dtype=data.dtype, stride=3)
                        self.addColumn("y", offset + bytesize, data.shape[0], dtype=data.dtype, stride=3)
                        self.addColumn("z", offset + bytesize * 2, data.shape[0], dtype=data.dtype, stride=3)
                    elif name == "Velocity":
                        offset = data.id.get_offset()
                        self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
                        self.addColumn("vy", offset + bytesize, data.shape[0], dtype=data.dtype, stride=3)
                        self.addColumn("vz", offset + bytesize * 2, data.shape[0], dtype=data.dtype, stride=3)
                    elif name == "Velocities":
                        offset = data.id.get_offset()
                        self.addColumn("vx", offset, data.shape[0], dtype=data.dtype, stride=3)
                        self.addColumn("vy", offset + bytesize, data.shape[0], dtype=data.dtype, stride=3)
                        self.addColumn("vz", offset + bytesize * 2, data.shape[0], dtype=data.dtype, stride=3)
                    else:
                        logger.error("unsupported column: %r of shape %r" % (name, array.shape))
        if "Header" in h5file:
            for name in "Redshift Time_GYR".split():
                if name in h5file["Header"].attrs:
                    value = h5file["Header"].attrs[name].decode("utf-8")
                    logger.debug("property[{name!r}] = {value}".format(**locals()))
                    self.variables[name] = value
                    # self.property_names.append(name)

        name = "particle_type"
        value = particle_type
        logger.debug("property[{name}] = {value}".format(**locals()))
        self.variables[name] = value
        # self.property_names.append(name)

    @classmethod
    def can_open(cls, path, fs_options={}, fs=None, *args, **kwargs):
        path = vaex.file.stringyfy(path)
        if len(args) == 2:
            particleName = args[0]
            particleType = args[1]
        elif "particle_name" in kwargs:
            particle_type = gadget_particle_names.index(kwargs["particle_name"].lower())
        elif "particle_type" in kwargs:
            particle_type = kwargs["particle_type"]
        elif "#" in path:
            path, index = path.split("#")
            particle_type = gadget_particle_names[index]
        else:
            return False
        h5file = None
        try:
            h5file = h5py.File(path, "r")
        except:
            return False
        has_particles = False
        # for i in range(1,6):
        key = "/PartType%d" % particle_type
        exists = key in h5file
        h5file.close()
        return exists

        # has_particles = has_particles or (key in h5file)
        # return has_particles

    @classmethod
    def get_options(cls, path):
        return []

    @classmethod
    def option_to_args(cls, option):
        return []


# dataset_type_map["gadget-hdf5"] = Hdf5MemoryMappedGadget

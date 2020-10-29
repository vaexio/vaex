import os

import astropy.io.fits as fits

from vaex.dataset_mmap import DatasetMemoryMapped


class FitsBinTable(DatasetMemoryMapped):
    def __init__(self, filename, write=False):
        super(FitsBinTable, self).__init__(filename, write=write)
        self.ucds = {}
        self.units = {}
        with fits.open(filename) as fitsfile:
            for table in fitsfile:
                if isinstance(table, fits.BinTableHDU):
                    table_offset = table._data_offset
                    #import pdb
                    #pdb.set_trace()
                    if table.columns[0].dim is not None: # for sure not a colfits
                        dim = eval(table.columns[0].dim) # TODO: can we not do an eval here? not so safe
                        if len(dim) == 2 and dim[0] <= dim[1]: # we have colfits format
                            logger.debug("colfits file!")
                            offset = table_offset
                            for i in range(len(table.columns)):
                                column = table.columns[i]
                                cannot_handle = False
                                column_name = column.name.strip()
                                self._get_column_meta_data(table, column_name, column, i)


                                # flatlength == length * arraylength
                                flatlength, fitstype = int(column.format[:-1]),column.format[-1]
                                arraylength, length = arrayshape = eval(column.dim)

                                # numpy dtype code, like f8, i4
                                dtypecode = astropy.io.fits.column.FITS2NUMPY[fitstype]


                                dtype = np.dtype((">" +dtypecode, arraylength))
                                if 0:
                                    if arraylength > 1:
                                        dtype = np.dtype((">" +dtypecode, arraylength))
                                    else:
                                        if dtypecode == "a": # I think numpy needs by default a length 1
                                            dtype = np.dtype(dtypecode + "1")
                                        else:
                                            dtype = np.dtype(">" +dtypecode)
                                    #	bytessize = 8

                                bytessize = dtype.itemsize
                                logger.debug("%r", (column.name, dtype, column.format, column.dim, length, bytessize, arraylength))
                                if (flatlength > 0): # and dtypecode != "a": # TODO: support strings
                                    if dtypecode == "a": # for ascii, we need to add the length again..
                                        dtypecode += str(arraylength)
                                    logger.debug("column type: %r", (column.name, offset, dtype, length, column.format, column.dim))
                                    if arraylength == 1 or dtypecode[0] == "a":
                                        ar = self._map_array(offset=offset, dtype=dtype, length=length)
                                        self.add_column(column_name, ar)
                                    else:
                                        for i in range(arraylength):
                                            name = column_name+"_" +str(i)
                                            self.addColumn(name, offset=offset+bytessize*i//arraylength, dtype=">" +dtypecode, length=length, stride=arraylength)
                                if flatlength > 0: # flatlength can be
                                    offset += bytessize * length
                                self._check_null(table, column_name, column, i)

                    else:
                        logger.debug("adding table: %r" % table)
                        for i, column in enumerate(table.columns):
                            array = column.array[:]
                            array = column.array[:] # 2nd time it will be a real np array
                            #import pdb
                            #pdb.set_trace()
                            if array.dtype.kind in "fiubSU":
                                column_name = _python_save_name(column.name, used=self.columns.keys())
                                self.addColumn(column_name, array=array)
                                self._get_column_meta_data(table, column_name, column, i)
                                self._check_null(table, column_name, column, i)
            self._try_votable(fitsfile[0])

        self._freeze()
        # self.update_meta()
        # self.update_virtual_meta()

    def _check_null(self, table, column_name, column, i):
        null_name = "TNULL%d" % (i+1)
        if null_name in table.header:
            mask_value = table.header[null_name]
            array = self._columns[column_name]
            mask = array == mask_value
            self._columns[column_name] = numpy.ma.masked_array(array, mask)

    def _try_votable(self, table):
        try:
            from io import BytesIO as StringIO
        except:
            from StringIO import StringIO
        if table.data is None:
            return
        vodata = table.data.tostring()
        if vodata.startswith(b"<?xml"):
            f = StringIO()
            f.write(vodata)
            votable = astropy.io.votable.parse(f)
            first_table = votable.get_first_table()
            used_names = []
            for field in first_table.fields:
                name = field.name.strip()
                clean_name = _python_save_name(name, used=used_names)
                used_names.append(name)
                if field.ucd:
                    self.ucds[clean_name] = field.ucd
                unit = _try_unit(field.unit)
                if unit:
                    self.units[clean_name] = unit
                if unit is None and field.unit:
                    print("unit error for: %r", field.unit)
                self.descriptions[clean_name] = field.description
            self.description = first_table.description

    def _get_column_meta_data(self, table, column_name, column, i):
        ucd_header_name = "TUCD%d" % (i+1)
        if ucd_header_name in table.header:
            self.ucds[column_name] = table.header[ucd_header_name]
        if column.unit:
            try:
                unit = _try_unit(column.unit)
                if unit:
                    self.units[column_name] = unit
            except:
                logger.exception("could not understand unit: %s" % column.unit)
        else: # we may want to try ourselves
            unit_header_name = "TUNIT%d" % (i+1)
            if unit_header_name in table.header:
                unit_str = table.header[unit_header_name]
                unit = _try_unit(unit_str)
                if unit:
                    self.unit[column_name] = unit
        #unit_header_name = "TUCD%d" % (i+1)
        #if ucd_header_name in table.header:
    @classmethod
    def can_open(cls, path, *args, **kwargs):
        return os.path.splitext(path)[1] == ".fits"

    @classmethod
    def get_options(cls, path):
        return [] # future: support multiple tables?

    @classmethod
    def option_to_args(cls, option):
        return []

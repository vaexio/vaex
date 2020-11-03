import numpy as np


from vaex.dataset import DatasetArrays


class DatasetTap(DatasetArrays):
    class TapColumn(object):
        def __init__(self, tap_dataset, column_name, column_type, ucd):
            self.tap_dataset = tap_dataset
            self.column_name = column_name
            self.column_type = column_type
            self.ucd = ucd
            self.alpha_min = 0
            length = len(tap_dataset)
            steps = length/1e6 # try to do it in chunks
            self.alpha_step = 360/steps
            self.alpha_max = self.alpha_min + self.alpha_step
            logger.debug("stepping in alpha %f" % self.alpha_step)
            self.data = []
            self.offset = 0
            self.shape = (length,)
            self.dtype = DatasetTap.type_map[self.column_type]().dtype
            self.left_over_chunk = None
            self.rows_left = length
            import tempfile
            self.download_file = tempfile.mktemp(".vot")

        def __getitem__(self, slice):
            start, stop, step = slice.start, slice.stop, slice.step
            required_length = stop - start
            assert start >= self.offset
            chunk_data = self.left_over_chunk
            enough = False if chunk_data is None else len(chunk_data) >= required_length
            if chunk_data is not None:
                logger.debug("start %s offset %s chunk length %s", start, self.offset, len(chunk_data))
                #assert len(chunk_data) == start - self.offset
            if enough:
                logger.debug("we can skip the query, already have results from previous query")
            while not enough:
                adql_query = "SELECT {column_name} FROM {table_name} WHERE alpha >= {alpha_min} AND alpha < {alpha_max} ORDER BY alpha ASC"\
                    .format(column_name=self.column_name, table_name=self.tap_dataset.table_name, alpha_min=self.alpha_min, alpha_max=self.alpha_max)
                logger.debug("executing: %s" % adql_query)
                logger.debug("executing: %s" % adql_query.replace(" ", "+"))


                url = self.tap_dataset.tap_url + "/sync?REQUEST=doQuery&LANG=ADQL&MAXREC=10000000&FORMAT=votable&QUERY=" +adql_query.replace(" ", "+")
                import urllib2
                response = urllib2.urlopen(url)
                with open(self.download_file, "w") as f:
                    f.write(response.read())
                votable = astropy.io.votable.parse(self.download_file)
                data = votable.get_first_table().array[self.column_name].data
                # TODO: respect masked array
                #table = astropy.table.Table.read(url, format="votable") #, show_progress=False)
                #data = table[self.column_name].data.data.data
                logger.debug("new chunk is of lenght %d", len(data))
                self.rows_left -= len(data)
                logger.debug("rows left %d", self.rows_left)
                if chunk_data is None:
                    chunk_data = data
                else:
                    chunk_data = np.concatenate([chunk_data, data])
                if len(chunk_data) >= required_length:
                    enough = True
                logger.debug("total chunk is of lenght %d, enough: %s", len(chunk_data), enough)
                self.alpha_min += self.alpha_step
                self.alpha_max += self.alpha_step


            result, self.left_over_chunk = chunk_data[:required_length], chunk_data[required_length:]
            #print(result)
            logger.debug("left over is of length %d", len(self.left_over_chunk))
            return result #np.zeros(N, dtype=self.dtype)



    type_map = {
        'REAL':np.float32,
        'SMALLINT':np.int32,
        'DOUBLE':np.float64,
        'BIGINT':np.int64,
        'INTEGER':np.int32,
        'BOOLEAN':np.bool8
    }
    #not supported types yet 'VARCHAR',', u'BOOLEAN', u'INTEGER', u'CHAR
    def __init__(self, tap_url="http://gaia.esac.esa.int/tap-server/tap/g10_smc", table_name=None):
        logger.debug("tap url: %r", tap_url)
        self.tap_url = tap_url
        self.table_name = table_name
        if table_name is None: # let us try to infer the table name
            if tap_url.endswith("tap") or tap_url.endswith("tap/"):
                pass # this mean we really didn't provide one
            else:
                index = tap_url.rfind("tap/")
                if index != -1:
                    self.tap_url, self.table_name = tap_url[:index+4], self.tap_url[index+4:]
                    logger.debug("inferred url is %s, and table name is %s", self.tap_url, self.table_name)

        if self.tap_url.startswith("tap+"): # remove tap+ part from tap+http(s), only keep http(s) part
            self.tap_url = self.tap_url[len("tap+"):]
        import requests
        super(DatasetTap, self).__init__(self.table_name)
        self.req = requests.request("get", self.tap_url+"/tables/")
        self.path = "tap+" +self.tap_url + "/" + table_name

        #print dir(self.req)
        from bs4 import BeautifulSoup
            #self.soup = BeautifulSoup(req.response)
        tables = BeautifulSoup(self.req.content, 'xml')
        self.tap_tables = collections.OrderedDict()
        for table in tables.find_all("table"):
            #print table.find("name").string, table.description.string, table["gaiatap:size"]
            table_name = unicode(table.find("name").string)
            table_size = int(table["esatapplus:size"])
            #print table_name, table_size
            logger.debug("tap table %r ", table_name)
            columns = []
            for column in table.find_all("column"):
                column_name = unicode(column.find("name").string)
                column_type = unicode(column.dataType.string)
                ucd = column.ucd.string if column.ucd else None
                unit = column.unit.string if column.unit else None
                description = column.description.string if column.description else None
                #print "\t", column_name, column_type, ucd
                #types.add()
                columns.append((column_name, column_type, ucd, unit, description))
            self.tap_tables[table_name] = (table_size, columns)
        if not self.tap_tables:
            raise ValueError("no tables or wrong url")
        for name, (table_size, columns) in self.tap_tables.items():
            logger.debug("table %s has length %d", name, table_size)
        self._full_length, self._tap_columns = self.tap_tables[self.table_name]
        self._length = self._full_length
        logger.debug("selected table table %s has length %d", self.table_name, self._full_length)
        #self.column_names = []
        #self.columns = collections.OrderedDict()
        for column_name, column_type, ucd, unit, description in self._tap_columns:
            logger.debug("  column %s has type %s and ucd %s, unit %s and description %s", column_name, column_type, ucd, unit, description)
            if column_type in self.type_map.keys():
                self.column_names.append(column_name)
                if ucd:
                    self.ucds[column_name] = ucd
                if unit:
                    self.units[column_name] = unit
                if description:
                    self.descriptions[column_name] = description
                self.columns[column_name] = self.TapColumn(self, column_name, column_type, ucd)
            else:
                logger.warning("  type of column %s is not supported, it will be skipped", column_name)


    @classmethod
    def open(cls, path, *args, **kwargs):
        return cls(path, *args, **kwargs)

    @classmethod
    def quick_test(cls, path, *args, **kwargs):
        return False

    @classmethod
    def can_open(cls, path, *args, **kwargs):
        can_open = False
        url = None
        try:
            url = urlparse(path)
        except:
            return False
        if url.scheme:
            if url.scheme.startswith("tap+http"): # will also catch https
                can_open = True
        logger.debug("%r can open: %r"  %(cls.__name__, can_open))
        return can_open

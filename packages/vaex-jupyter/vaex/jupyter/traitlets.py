from __future__ import absolute_import
import traitlets

def nice_type(df, name):
    dtype = df.dtype(name)
    type_map = {'i': 'integer', 'u': 'integer', 'f': 'float', 'b': 'boolean',
               'M': 'date/time'}
    if dtype == str:
        return 'string'
    else:
        return type_map[dtype.kind]


class ColumnsMixin(traitlets.HasTraits):
    df = traitlets.Any().tag(readonly=True)
    columns = traitlets.Any([
        {
            'name': 'foo',
            'virtual': False,
            'dtype': 'string'
        },
        {
            'name': 'bar',
            'virtual': True,
            'dtype': 'string',
            'expression': 'foo + "spam"'
        }
    ]).tag(sync=True)
    def __init__(self, df=None, **kwargs):
        super(ColumnsMixin, self).__init__(df=df, **kwargs)
        if df:
            df.signal_column_changed.connect(self._on_column_changed)
            self._populate_columns()

    def _on_column_changed(self, df, name, action):
        assert df == self.df
        self._populate_columns()

    def _populate_columns(self):
        def _():
            for column_name in self.df.get_column_names():
                expression = self.df.virtual_columns.get(column_name)
                item = {
                    'name': column_name,
                    'dtype': nice_type(self.df, column_name),
                    'virtual': expression is not None
                }
                if item['virtual']:
                    item['expression'] = expression
                yield item
        self.columns = list(_())

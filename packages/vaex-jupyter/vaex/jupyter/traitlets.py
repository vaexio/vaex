import traitlets
import vaex


def nice_type(df, name):
    dtype = df.data_type(name)
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


class Expression(traitlets.Any):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag(to_json=self._to_json, from_json=self._from_json)

    def _to_json(self, value, widget):
        return str(value)

    def _from_json(self, value, widget):
        return widget.df[value]

    def validate(self, obj, value):
        try:
            df = obj.df  # we assume the object has a dataframe instance
        except traitlets.traitlets.TraitError:
            if value is None:
                return None
            else:
                raise
        if isinstance(value, str):
            df.validate_expression(value)
            return df[value]
        elif isinstance(value, vaex.expression.Expression):
            return value
        else:
            raise traitlets.TraitError(f'{value} should be a string or vaex expression')


def traitlet_fixes(cls):
    """Applies all vaex opinionated traitlet fixes"""
    return patch_trait_docstrings(cls)


def patch_trait_docstrings(cls):
    """Put the help string as docstring in all traits of a class"""
    for trait_name, trait in cls.class_traits().items():
        if 'help' in trait.metadata:
            trait.__doc__ = trait.metadata['help']
    return cls

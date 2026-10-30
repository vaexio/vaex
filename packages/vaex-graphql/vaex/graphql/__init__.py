import graphene
import graphql.execution

import vaex
import pandas as pd


class DataFrameAccessorGraphQL(object):
    """Exposes a GraphQL layer to a DataFrame

    See `the GraphQL example
    <http://docs.vaex.io/en/latest/example_graphql.html>`_ for more usage.

    The easiest way to learn to use the GraphQL language/vaex interface is to
    launch a server, and play with the GraphiQL graphical interface, its
    autocomplete, and the schema explorer.

    We try to stay close to the Hasura API: https://docs.hasura.io/1.0/graphql/manual/api-reference/graphql-api/query.html
    """
    def __init__(self, df):
        self.df = df

    def query(self, name='df'):
        """Creates a graphene query object exposing this DataFrame named `name`"""
        return create_query({name: self.df})
  
    def schema(self, name='df', auto_camelcase=False, **kwargs):
        """Creates a graphene schema for this DataFrame"""
        return graphene.Schema(query=self.query(name), auto_camelcase=auto_camelcase, **kwargs)
    
    def execute(self, *args, **kwargs):
        """Creates a schema, and execute the query (first argument)
        """
        return self.schema().execute(*args, **kwargs)

    def serve(self, port=9001, address='', name='df', verbose=True):
        """Serve the DataFrame via a http server"""
        from .tornado import Application
        schema = self.schema(name=name)
        app = Application(schema)
        app.listen(port, address)
        if not address:
            address = 'localhost'
        if verbose:
            print('serving at: http://{address}:{port}/graphql'.format(**locals()))


@pd.api.extensions.register_dataframe_accessor("graphql")
class DataFrameAccessorGraphQLPandas(DataFrameAccessorGraphQL):
    def __init__(self, df):
        super(DataFrameAccessorGraphQLPandas, self).__init__(vaex.from_pandas(df))


def map_to_field(df, name):
    dtype = df[name].data_type()
    if vaex.array_types.is_string_type(dtype):
        return graphene.String
    elif dtype.kind == "f":
        return graphene.Float
    elif dtype.kind == "i":
        return graphene.Int
    elif dtype.kind == "b":
        return graphene.Boolean
    elif dtype.kind == "M":
        return graphene.DateTime
    elif dtype.kind == "m":
        return graphene.DateTime  # TODO, not sure how we're gonna deal with timedelta
    else:
        raise ValueError('dtype not supported: %r' % dtype)


class Compare(graphene.InputObjectType):
    def filter(self, df, name):
        expression = vaex.expression.Expression(df, '(1==1)')
        if self._eq is not None:
            expression = expression & (df[name] == self._eq)
        if self._neq is not None:
            expression = expression & (df[name] != self._neq)
        return expression


class NumberCompare(Compare):
    def filter(self, df, name):
        expression = super(NumberCompare, self).filter(df, name)
        if self._gt is not None:
            expression = expression & (df[name] > self._gt)
        if self._lt is not None:
            expression = expression & (df[name] < self._lt)
        if self._gte is not None:
            expression = expression & (df[name] >= self._gte)
        if self._lte is not None:
            expression = expression & (df[name] <= self._lte)
        return expression

class IntCompare(NumberCompare):
    _eq = graphene.Field(graphene.Int)
    _neq = graphene.Field(graphene.Int)
    _gt = graphene.Field(graphene.Int)
    _lt = graphene.Field(graphene.Int)
    _gte = graphene.Field(graphene.Int)
    _lte = graphene.Field(graphene.Int)

class FloatCompare(NumberCompare):
    _eq = graphene.Field(graphene.Float)
    _neq = graphene.Field(graphene.Float)
    _gt = graphene.Field(graphene.Float)
    _lt = graphene.Field(graphene.Float)
    _gte = graphene.Field(graphene.Float)
    _lte = graphene.Field(graphene.Float)

class BooleanCompare(Compare):
    _eq = graphene.Field(graphene.Boolean)
    _neq = graphene.Field(graphene.Boolean)

class StringCompare(Compare):
    _eq = graphene.Field(graphene.String)
    _neq = graphene.Field(graphene.String)

class DateTimeCompare(Compare):
    _eq = graphene.Field(graphene.String)
    _neq = graphene.Field(graphene.String)

def map_to_comparison(df, name):
    dtype = df[name].data_type()
    if vaex.array_types.is_string_type(dtype):
        return StringCompare
    elif dtype.kind == "f":
        return FloatCompare
    elif dtype.kind == "i":
        return IntCompare
    elif dtype.kind == "b":
        return BooleanCompare
    elif dtype.kind == "M":
        return DateTimeCompare
    elif dtype.kind == "m":
        return DateTimeCompare  # TODO, not sure how we're gonna deal with timedelta
    else:
        raise ValueError('dtype not supported: %r' % dtype)

def create_aggregation_on_field(df, groupby):
    postfix = "_".join(groupby)
    if postfix:
        postfix = "_" + postfix
    class AggregationOnFieldBase(graphene.ObjectType):
        def __init__(self, df, agg, groupby):
            self.df = df
            self.agg = agg
            self.groupby = groupby
    class Meta:
        def default_resolver(name, __, obj, info):
            if obj.groupby:
                groupby = obj.df.groupby(obj.groupby)
                agg = getattr(vaex.agg, obj.agg)(name)
                dfg = groupby.agg({'agg': agg})
                agg_values = dfg['agg']
                return agg_values.tolist()
            return getattr(obj.df, obj.agg)(name)
    if groupby:
        attrs = {name: graphene.List(map_to_field(df, name)) for name in df.get_column_names()}
    else:
        attrs = {name: map_to_field(df, name)() for name in df.get_column_names()}
    attrs['Meta'] = Meta
    DataFrameAggregationOnField = type("AggregationOnField"+postfix, (AggregationOnFieldBase, ), attrs)
    return DataFrameAggregationOnField

    
def create_groupby(df, groupby):
    postfix = "_".join(groupby)
    if postfix:
        postfix = "_" + postfix
    class GroupByBase(graphene.ObjectType):
        count = graphene.List(graphene.Int)
        keys = graphene.List(graphene.Int)

        def __init__(self, df, by=None):
            self.df = df
            self.by = [] if by is None else by
            self._groupby = None # cached lazy object
            super(GroupByBase, self).__init__()

        @property
        def groupby(self):
            if self._groupby is None:
                self._groupby = self.df.groupby(self.by)
            return self._groupby

        def resolve_count(self, info):
            dfg = self.groupby.agg('count')
            return dfg['count'].tolist()

        def resolve_keys(self, info):
            return self.groupby.coords1d[-1]

    def field_groupby(name):
        Aggregate = create_aggregate(df, groupby + [name])
        def resolver(*args, **kwargs):
            return Aggregate(df, groupby + [name])
        return graphene.Field(Aggregate, resolver=resolver)
    attrs = {name: field_groupby(name) for name in df.get_column_names()}
    GroupBy = type("GroupBy"+postfix, (GroupByBase, ), attrs)
    return GroupBy


def create_row(df):
    class RowBase(graphene.ObjectType):
        pass
    attrs = {name: map_to_field(df, name)() for name in df.get_column_names()}
    Row = type("Row", (RowBase, ), attrs)
    return Row

def create_aggregate(df, groupby=None):
    postfix = "_".join(groupby)
    if postfix:
        postfix = "_" + postfix
    if groupby is None: groupby = []
    
    AggregationOnField = create_aggregation_on_field(df, groupby)
    if len(groupby):
        CountType = graphene.List(graphene.Int)
        
    else:
        CountType = graphene.Int()
        
    Row = create_row(df)

    class AggregationBase(graphene.ObjectType):
        count = CountType
        min = graphene.Field(AggregationOnField)
        max = graphene.Field(AggregationOnField)
        mean = graphene.Field(AggregationOnField)
        if not groupby:
            row = graphene.List(Row, limit=graphene.Int(default_value=100), offset=graphene.Int())

        def __init__(self, df, by=None):
            self.df = df
            self.by = [] if by is None else by
            assert self.by == groupby
            self._groupby = None # cached lazy object
            super(AggregationBase, self).__init__()

        if not groupby:
            def resolve_row(self, info, limit, offset=None):
                fields = [field for field in info.field_asts if field.name.value == info.field_name]
                subfields = fields[0].selection_set.selections
                names = [subfield.name.value for subfield in subfields]
                if offset is None:
                    offset = 0
                N = min(len(self.df)-offset, limit)
                df_pagination = self.df[offset:offset+N]
                matrix = [df_pagination[name].tolist() for name in names]
                rows = []
                for i in range(len(df_pagination)):
                    row = Row()
                    for j, name in enumerate(names):
                        setattr(row, name, matrix[j][i])
                    rows.append(row)
                return rows

        @property
        def groupby_object(self):
            if self._groupby is None:
                self._groupby = self.df.groupby(self.by)
            return self._groupby

        def resolve_count(self, info):
            if self.by:
                groupby = self.groupby_object
                dfg = self.groupby_object.agg('count')
                counts = dfg['count']
                return counts.tolist()
            return len(self.df)
        def resolve_min(self, info):
            return AggregationOnField(self.df, 'min', self.by)
        def resolve_max(self, info):
            return AggregationOnField(self.df, 'max', self.by)
        def resolve_mean(self, info):
            return AggregationOnField(self.df, 'mean', self.by)

    attrs = {}
    if len(groupby) < 2:
        GroupBy = create_groupby(df, groupby)
        def resolver(*args, **kwargs):
            return GroupBy(df, groupby)
        attrs["groupby"] = graphene.Field(GroupBy, resolver=resolver, description="GroupBy a field for %s" % '...')

    Aggregation = type("Aggregation"+postfix, (AggregationBase, ), attrs)
    return Aggregation


# similar to https://docs.hasura.io/1.0/graphql/manual/api-reference/graphql-api/query.html#whereexp
def create_boolexp(df):
    column_names = df.get_column_names()
    class BoolExpBase(graphene.InputObjectType):
        _and = graphene.List(lambda: BoolExp)
        _or = graphene.List(lambda: BoolExp)
        _not = graphene.Field(lambda: BoolExp)

        def filter(self, df):
            expression = vaex.expression.Expression(df, '(1==1)')
            if self._and:
                for value in self._and:
                    expression = expression & value.filter(df)
            if self._or:
                or_expression = self._or[0].filter(df)
                for value in self._or[1:]:
                    or_expression = or_expression | value.filter(df)
                expression = expression & or_expression
            if self._not:
                expression = expression & ~self._not.filter(df)

            for name in column_names:
                value = getattr(self, name)
                if value is not None:
                    expression = expression & value.filter(df, name)
            return expression

    attrs = {}
    for name in column_names:
        attrs[name] = graphene.Field(map_to_comparison(df, name))
    BoolExp = type("BoolExp", (BoolExpBase, ), attrs)
    return BoolExp

def create_query(dfs):
    class QueryBase(graphene.ObjectType):
        hello = graphene.String(description='A typical hello world')
    fields = {}
    for name, df in dfs.items():
        columns = df.get_column_names()
        columns = [k for k in columns if df.is_string(k) or df[k].dtype.kind != 'O']
        df = df[columns]
        Aggregate = create_aggregate(df, [])
        def closure(df=df, name=name, Aggregate=Aggregate):
            def resolve(parent, info, where=None):
                if where is not None:
                    expression = where.filter(df)
                    if expression.expression:
                        return Aggregate(df=df[expression])
                return Aggregate(df=df)
            return resolve
        Where = create_boolexp(df)
        fields[name] = graphene.Field(Aggregate, resolver=closure(), description="Aggregations for %s" % name, where=graphene.Argument(Where))
        
    return type("Query", (QueryBase, ), fields)

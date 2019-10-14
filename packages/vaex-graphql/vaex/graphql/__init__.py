import graphene
import graphql.execution
import vaex

if graphene.VERSION >= (3, 0):
    class ExecutionContext(graphql.execution.ExecutionContext):
        def __init__(self, *args, **kwargs):
            self.call_stack_count = 0
            super().__init__(*args, **kwargs)

        def execute_fields(
            self,
            parent_type,
            source_value,
            path,
            fields,
        ):
            try:
                self.call_stack_count += 1
                result = super().execute_fields(parent_type, source_value, path, fields)
            finally:
                self.call_stack_count -= 1
            if self.call_stack_count == 0:
                for df in self.context_value['dataframes']:
                    df.execute()
            return result
else:
    ExecutionContext = None


class DataFrameAccessorGraphQL(object):
    def __init__(self, df):
        self.df = df

    def query(self, name='df'):
        """Creates a graphene query object exposing this dataframe named `name`"""
        return create_query({name: self.df})
  
    def schema(self, name='df', auto_camelcase=False, **kwargs):
        return graphene.Schema(query=self.query(name), auto_camelcase=auto_camelcase, **kwargs)
    
    def execute(self, *args, **kwargs):
        if ExecutionContext:
            if 'execution_context_class' not in kwargs:
                kwargs['execution_context_class'] = ExecutionContext
                assert 'context_value' not in kwargs
                kwargs['context_value'] = {'dataframes': [self.df]}
        return self.schema().execute(*args, **kwargs)

    def serve(self, port=9001, address='', name='df', verbose=True):
        from .tornado import Application
        schema = self.schema(name=name)
        app = Application(schema)
        app.listen(port, address)
        if not address:
            address = 'localhost'
        if verbose:
            print(f'serving at: http://{address}:{port}/graphql')


def map_to_field(df, name):
    dtype = df[name].dtype
    if dtype == str:
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

# def create_groupedby(df, groupby):
#     postfix = "_".join(groupby)
#     if postfix:
#         posfix = "_" + postfix
#     class AggregationOnFieldBase(graphene.ObjectType):
#         def __init__(self, df, agg, groupby):
#             self.df = df
#             self.agg = agg
#             self.groupby = groupby
#     class Meta:
#         def default_resolver(name, __, obj, info):
#             return getattr(obj.df, obj.agg)(name)
#     attrs = {name: map_to_field(df, name) for name in df.get_column_names()}
#     attrs['Meta'] = Meta
#     DataFrameAggregationOnField = type("AggregationOnField"+postfix, (AggregationOnFieldBase, ), attrs)
#     return DataFrameAggregationOnField
    
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

#         class Meta:
#         def default_resolver(name, __, obj, info):
#             return getattr(obj.df, obj.agg)(name)
    def field_groupby(name):
        Aggregate = create_aggregate(df, groupby + [name])
        def resolver(*args, **kwargs):
            return Aggregate(df, groupby + [name])
        return graphene.Field(Aggregate, resolver=resolver)
    attrs = {name: field_groupby(name) for name in df.get_column_names()}
#     attrs['Meta'] = Meta
    GroupBy = type("GroupBy"+postfix, (GroupByBase, ), attrs)
    return GroupBy
    
def create_aggregate(df, groupby=None):
    postfix = "_".join(groupby)
    if postfix:
        postfix = "_" + postfix
    if groupby is None: groupby = []
    
    AggregationOnField = create_aggregation_on_field(df, groupby)
    if len(groupby):
#         CountType = graphene.Int
        CountType = graphene.List(graphene.Int)
        
    else:
        CountType = graphene.Int()
#     for i in range(len(groupby)):
#         CountType = graphene.List(CountType)
        

    class AggregationBase(graphene.ObjectType):
        count = CountType
        min = graphene.Field(AggregationOnField)
        max = graphene.Field(AggregationOnField)
        mean = graphene.Field(AggregationOnField)

        hello = graphene.String(description='A typical hello world')
        def resolve_hello(self, info):
            return 'World'

        def __init__(self, df, by=None):
            self.df = df
            self.by = [] if by is None else by
            assert self.by == groupby
            self._groupby = None # cached lazy object
            super(AggregationBase, self).__init__()

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

    # fields_agg = {name: graphene.Field(DataFrameAggregationField) for name in df.get_column_names()}
    attrs = {}
    if len(groupby) < 2:
        GroupBy = create_groupby(df, groupby)
        def resolver(*args, **kwargs):
            return GroupBy(df, groupby)
        attrs["groupby"] = graphene.Field(GroupBy, resolver=resolver, description="GroupBy a field for %s" % '...')

    Aggregation = type("Aggregation"+postfix, (AggregationBase, ), attrs)
    return Aggregation

def create_query(dfs):
    class QueryBase(graphene.ObjectType):
        hello = graphene.String(description='A typical hello world')
    fields = {}
    for name, df in dfs.items():
        columns = df.get_column_names()
        columns = [k for k in columns if df[k].dtype == str or df[k].dtype.kind != 'O']
        df = df[columns]
        Aggregate = create_aggregate(df, [])
        def closure(df=df, name=name, Aggregate=Aggregate):
            def resolve(*args, **kwargs):
                return Aggregate(df=df)
            return resolve
        fields[name] = graphene.Field(Aggregate, resolver=closure(), description="Aggregations for %s" % name)
        
#         GroupBy = create_groupby(df, [])
#         def closure(df=df, name=name, GroupBy=GroupBy):
#             def resolve(*args, **kwargs):
#                 return GroupBy(df=df)
#             return resolve
#         fields[name+"_groupby"] = graphene.Field(GroupBy, resolver=closure(), description="GroupBy a field for %s" % name)
    return type("Query", (QueryBase, ), fields)
# schema = graphene.Schema(query=create_query({'taxi': df}), auto_camelcase=False)
# schema
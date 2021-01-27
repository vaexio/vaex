import ast
import os
import base64
import cloudpickle as pickle
import functools
import operator
import six
import collections
import weakref

from future.utils import with_metaclass
import numpy as np
import tabulate
import pyarrow as pa
from vaex.datatype import DataType
from vaex.docstrings import docsubst

from vaex.utils import _ensure_strings_from_expressions, _ensure_string_from_expression
from vaex.column import ColumnString, _to_string_sequence
from .hash import counter_type_from_dtype
import vaex.serialize
from . import expresso


try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections


# TODO: repeated from dataframe.py
default_shape = 128
PRINT_MAX_COUNT = 10

expression_namespace = {}
expression_namespace['nan'] = np.nan


expression_namespace = {}
expression_namespace['nan'] = np.nan


_binary_ops = [
    dict(code="+", name='add', op=operator.add),
    dict(code="in", name='contains', op=operator.contains),
    dict(code="/", name='truediv', op=operator.truediv),
    dict(code="//", name='floordiv', op=operator.floordiv),
    dict(code="&", name='and', op=operator.and_),
    dict(code="^", name='xor', op=operator.xor),

    dict(code="|", name='or', op=operator.or_),
    dict(code="**", name='pow', op=operator.pow),
    dict(code="is", name='is', op=operator.is_),
    dict(code="is not", name='is_not', op=operator.is_not),

    dict(code="<<", name='lshift', op=operator.lshift),
    dict(code="%", name='mod', op=operator.mod),
    dict(code="*", name='mul', op=operator.mul),

    dict(code=">>", name='rshift', op=operator.rshift),
    dict(code="-", name='sub', op=operator.sub),

    dict(code="<", name='lt', op=operator.lt),
    dict(code="<=", name='le', op=operator.le),
    dict(code="==", name='eq', op=operator.eq),
    dict(code="!=", name='ne', op=operator.ne),
    dict(code=">=", name='ge', op=operator.ge),
    dict(code=">", name='gt', op=operator.gt),
]
if hasattr(operator, 'div'):
    _binary_ops.append(dict(code="/", name='div', op=operator.div))
if hasattr(operator, 'matmul'):
    _binary_ops.append(dict(code="@", name='matmul', op=operator.matmul))

reversable = 'add sub mul matmul truediv floordiv mod divmod pow lshift rshift and xor or'.split()

_unary_ops = [
    dict(code="~", name='invert', op=operator.invert),
    dict(code="-", name='neg', op=operator.neg),
    dict(code="+", name='pos', op=operator.pos),
]


class Meta(type):
    def __new__(upperattr_metaclass, future_class_name,
                future_class_parents, attrs):
        # attrs = {}
        for op in _binary_ops:
            def wrap(op=op):
                def f(a, b):
                    self = a
                    # print(op, a, b)
                    try:
                        stringy = isinstance(b, str) or b.is_string()
                    except:
                        # this can happen when expression is a literal, like '1' (used in propagate_unc)
                        # which causes the dtype to fail
                        stringy = False
                    if stringy:
                        if isinstance(b, str):
                            b = repr(b)
                        if op['code'] == '==':
                            expression = 'str_equals({0}, {1})'.format(a.expression, b)
                        elif op['code'] == '+':
                            expression = 'str_cat({0}, {1})'.format(a.expression, b)
                        else:
                            raise ValueError('operand %r not supported for string comparison' % op['code'])
                        return Expression(self.ds, expression=expression)
                    else:
                        if isinstance(b, Expression):
                            assert b.ds == a.ds
                            b = b.expression
                        elif isinstance(b, (np.timedelta64)):
                            df = a.ds
                            b = df.add_variable('var_time_delta', b, unique=True)
                        elif isinstance(b, (np.datetime64)):
                            df = a.ds
                            b = df.add_variable('var_date_time', b, unique=True)
                        expression = '({0} {1} {2})'.format(a.expression, op['code'], b)
                    return Expression(self.ds, expression=expression)
                attrs['__%s__' % op['name']] = f
                if op['name'] in reversable:
                    def f(a, b):
                        self = a
                        if isinstance(b, str):
                            if op['code'] == '+':
                                expression = 'str_cat({1}, {0})'.format(a.expression, repr(b))
                            else:
                                raise ValueError('operand %r not supported for string comparison' % op['code'])
                            return Expression(self.ds, expression=expression)
                        else:
                            if isinstance(b, Expression):
                                assert b.ds == a.ds
                                b = b.expression
                            expression = '({2} {1} {0})'.format(a.expression, op['code'], b)
                            return Expression(self.ds, expression=expression)
                    attrs['__r%s__' % op['name']] = f

            wrap(op)
        for op in _unary_ops:
            def wrap(op=op):
                def f(a):
                    self = a
                    expression = '{0}({1})'.format(op['code'], a.expression)
                    return Expression(self.ds, expression=expression)
                attrs['__%s__' % op['name']] = f
            wrap(op)
        return type(future_class_name, future_class_parents, attrs)


class DateTime(object):
    """DateTime operations

    Usually accessed using e.g. `df.birthday.dt.dayofweek`
    """
    def __init__(self, expression):
        self.expression = expression


class TimeDelta(object):
    """TimeDelta operations

    Usually accessed using e.g. `df.delay.td.days`
    """
    def __init__(self, expression):
        self.expression = expression


class StringOperations(object):
    """String operations.

    Usually accessed using e.g. `df.name.str.lower()`
    """
    def __init__(self, expression):
        self.expression = expression


class StringOperationsPandas(object):
    """String operations using Pandas Series (much slower)"""
    def __init__(self, expression):
        self.expression = expression


class Expression(with_metaclass(Meta)):
    """Expression class"""
    def __init__(self, ds, expression, ast=None):
        self.ds = ds
        assert not isinstance(ds, Expression)
        if isinstance(expression, Expression):
            expression = expression.expression
        self._ast = ast
        self._expression = expression
        self.df._expressions.append(weakref.ref(self))
        self._ast_names = None

    def copy(self, df=None):
        """Efficiently copies an expression.

        Expression objects have both a string and AST representation. Creating
        the AST representation involves parsing the expression, which is expensive.

        Using copy will deepcopy the AST when the expression was already parsed.

        :param df: DataFrame for which the expression will be evaluated (self.df if None)
        """
        # expression either has _expression or _ast not None
        if df is None:
            df = self.df
        if self._expression is not None:
            expression = Expression(df, self._expression)
            if self._ast is not None:
                expression._ast = copy.deepcopy(self._ast)
        elif self._ast is not None:
            expression = Expression(df, copy.deepcopy(self._ast))
            if self._ast is not None:
                expression._ast = self._ast
        return expression

    @property
    def ast(self):
        """Returns the abstract syntax tree (AST) of the expression"""
        if self._ast is None:
            self._ast = expresso.parse_expression(self.expression)
        return self._ast

    @property
    def ast_names(self):
        if self._ast_names is None:
            self._ast_names = expresso.names(self.ast)
        return self._ast_names

    @property
    def _ast_slices(self):
        return expresso.slices(self.ast)

    @property
    def expression(self):
        if self._expression is None:
            self._expression = expresso.node_to_string(self.ast)
        return self._expression

    @expression.setter
    def expression(self, value):
        # if we reassign to expression, we clear the ast cache
        if value != self._expression:
            self._expression = value
            self._ast = None

    def __bool__(self):
        """Cast expression to boolean. Only supports (<expr1> == <expr2> and <expr1> != <expr2>)

        The main use case for this is to support assigning to traitlets. e.g.:

        >>> bool(expr1 == expr2)

        This will return True when expr1 and expr2 are exactly the same (in string representation). And similarly for:

        >>> bool(expr != expr2)

        All other cases will return True.
        """
        # this is to make traitlets detect changes
        import _ast
        if isinstance(self.ast, _ast.Compare) and len(self.ast.ops) == 1 and isinstance(self.ast.ops[0], _ast.Eq):
            return expresso.node_to_string(self.ast.left) == expresso.node_to_string(self.ast.comparators[0])
        if isinstance(self.ast, _ast.Compare) and len(self.ast.ops) == 1 and isinstance(self.ast.ops[0], _ast.NotEq):
            return expresso.node_to_string(self.ast.left) != expresso.node_to_string(self.ast.comparators[0])
        return True


    @property
    def df(self):
        # lets gradually move to using .df
        return self.ds

    @property
    def dtype(self):
        return self.df.data_type(self.expression)

    # TODO: remove this method?
    def data_type(self, array_type=None, axis=0):
        return self.df.data_type(self.expression, axis=axis)

    @property
    def shape(self):
        return self.df._shape_of(self)

    def to_arrow(self, convert_to_native=False):
        '''Convert to Apache Arrow array (will byteswap/copy if convert_to_native=True).'''
        values = self.values
        return vaex.array_types.to_arrow(values, convert_to_native=convert_to_native)

    def __arrow_array__(self, type=None):
        values = self.to_arrow()
        return pa.array(values, type=type)

    def to_numpy(self, strict=True):
        """Return a numpy representation of the data"""
        values = self.values
        return vaex.array_types.to_numpy(values, strict=strict)

    def to_dask_array(self, chunks="auto"):
        import dask.array as da
        import uuid
        dtype = self.dtype
        chunks = da.core.normalize_chunks(chunks, shape=self.shape, dtype=dtype.numpy)
        name = 'vaex-expression-%s' % str(uuid.uuid1())
        def getitem(df, item):
            assert len(item) == 1
            item = item[0]
            start, stop, step = item.start, item.stop, item.step
            assert step in [None, 1]
            return self.evaluate(start, stop, parallel=False)
        dsk = da.core.getem(name, chunks, getitem=getitem, shape=self.shape, dtype=dtype.numpy)
        dsk[name] = self
        return da.Array(dsk, name, chunks, dtype=dtype.numpy)

    def to_pandas_series(self):
        """Return a pandas.Series representation of the expression.

        Note: Pandas is likely to make a memory copy of the data.
        """
        import pandas as pd
        return pd.Series(self.values)

    def __getitem__(self, slice):
        return self.ds[slice][self.expression]

    def __abs__(self):
        """Returns the absolute value of the expression"""
        return self.abs()

    @property
    def dt(self):
        """Gives access to datetime operations via :py:class:`DateTime`"""
        return DateTime(self)

    @property
    def td(self):
        """Gives access to timedelta operations via :py:class:`TimeDelta`"""
        return TimeDelta(self)

    @property
    def str(self):
        """Gives access to string operations via :py:class:`StringOperations`"""
        return StringOperations(self)

    @property
    def str_pandas(self):
        """Gives access to string operations via :py:class:`StringOperationsPandas` (using Pandas Series)"""
        return StringOperationsPandas(self)

    @property
    def values(self):
        return self.evaluate()

    def derivative(self, var, simplify=True):
        var = _ensure_string_from_expression(var)
        return self.__class__(self.ds, expresso.derivative(self.ast, var, simplify=simplify))

    def expand(self, stop=[]):
        """Expand the expression such that no virtual columns occurs, only normal columns.

        Example:

        >>> df = vaex.example()
        >>> r = np.sqrt(df.data.x**2 + df.data.y**2)
        >>> r.expand().expression
        'sqrt(((x ** 2) + (y ** 2)))'

        """
        stop = _ensure_strings_from_expressions(stop)
        def translate(id):
            if id in self.ds.virtual_columns and id not in stop:
                return self.ds.virtual_columns[id]
        expr = expresso.translate(self.ast, translate)
        return Expression(self.ds, expr)

    def variables(self, ourself=False, expand_virtual=True, include_virtual=True):
        """Return a set of variables this expression depends on.

        Example:

        >>> df = vaex.example()
        >>> r = np.sqrt(df.data.x**2 + df.data.y**2)
        >>> r.variables()
        {'x', 'y'}
        """
        variables = set()
        def record(varname):
            # do this recursively for virtual columns
            if varname in self.ds.virtual_columns and varname not in variables:
                if (include_virtual and (varname != self.expression)) or (varname == self.expression and ourself):
                    variables.add(varname)
                if expand_virtual:
                    variables.update(self.df[self.df.virtual_columns[varname]].variables(ourself=include_virtual, include_virtual=include_virtual))
            # we usually don't want to record ourself
            elif varname != self.expression or ourself:
                variables.add(varname)

        expresso.translate(self.ast, record)
        # df is a buildin, don't record it, if df is a column name, it will be collected as
        # df['df']
        variables -= {'df'}
        for varname in self._ast_slices:
            if varname in self.df.virtual_columns and varname not in variables:
                if (include_virtual and (f"df['{varname}']" != self.expression)) or (f"df['{varname}']" == self.expression and ourself):
                    variables.add(varname)
                if expand_virtual:
                    if varname in self.df.virtual_columns:
                        variables |= self.df[self.df.virtual_columns[varname]].variables(ourself=include_virtual, include_virtual=include_virtual)
            elif f"df['{varname}']" != self.expression or ourself:
                variables.add(varname)

        return variables

    def _graph(self):
        """"Return a graph containing the dependencies of this expression
        Structure is:
            [<string expression>, <function name if callable>, <function object if callable>, [subgraph/dependencies, ....]]
        """
        expression = self.expression

        def walk(node):
            if isinstance(node, six.string_types):
                if node in self.ds.virtual_columns:
                    ex = Expression(self.ds, self.ds.virtual_columns[node])
                    return [node, None, None, [ex._graph()]]
                else:
                    return node
            else:
                fname, node_repr, deps = node
                if len(node_repr) > 30:  # clip too long expressions
                    node_repr = node_repr[:26] + ' ....'
                deps = [walk(dep) for dep in deps]
                obj = self.ds.functions.get(fname)
                # we don't want the wrapper, we want the underlying object
                if isinstance(obj, Function):
                    obj = obj.f
                if isinstance(obj, FunctionSerializablePickle):
                    obj = obj.f
                return [node_repr, fname, obj, deps]
        return walk(expresso._graph(expression))

    def _graphviz(self, dot=None):
        """Return a graphviz.Digraph object with a graph of the expression"""
        from graphviz import Graph, Digraph
        node = self._graph()
        dot = dot or Digraph(comment=self.expression)
        def walk(node):
            if isinstance(node, six.string_types):
                dot.node(node, node)
                return node, node
            else:
                node_repr, fname, fobj, deps = node
                node_id = node_repr
                dot.node(node_id, node_repr)
                for dep in deps:
                    dep_id, dep = walk(dep)
                    dot.edge(node_id, dep_id)
                return node_id, node
        walk(node)
        return dot

    def __str__(self):
        return self.expression

    # def __array__(self, dtype=None):
    #     '''For casting to a numpy array

    #     Example:
    #         >>> np.array(ds.x**2)

    #     '''
    #     return self.ds.evaluate(self)

    def tolist(self, i1=None, i2=None):
        '''Short for expr.evaluate().tolist()'''
        values = self.evaluate(i1=i1, i2=i2)
        if isinstance(values, (pa.Array, pa.ChunkedArray)):
            return values.to_pylist()
        return values.tolist()

    if not os.environ.get('VAEX_DEBUG', ''):
        def __repr__(self):
            return self._repr_plain_()

    def _repr_plain_(self):
        from .formatting import _format_value
        def format(values):
            for i in range(len(values)):
                value = values[i]
                yield _format_value(value)
        colalign = ("right",) * 2
        try:
            N = len(self.ds)
            if N <= PRINT_MAX_COUNT:
                values = format(self.evaluate(0, N))
                values = tabulate.tabulate([[i, k] for i, k in enumerate(values)], tablefmt='plain', colalign=colalign)
            else:
                values_head = format(self.evaluate(0, PRINT_MAX_COUNT//2))
                values_tail = format(self.evaluate(N - PRINT_MAX_COUNT//2, N))
                values_head = list(zip(range(PRINT_MAX_COUNT//2), values_head)) +\
                              list(zip(range(N - PRINT_MAX_COUNT//2, N), values_tail))
                values = tabulate.tabulate([k for k in values_head], tablefmt='plain', colalign=colalign)
                values = values.split('\n')
                width = max(map(len, values))
                separator = '\n' + '...'.center(width, ' ') + '\n'
                values = "\n".join(values[:PRINT_MAX_COUNT//2]) + separator + "\n".join(values[PRINT_MAX_COUNT//2:]) + '\n'
        except Exception as e:
            values = 'Error evaluating: %r' % e
        expression = self.expression
        if len(expression) > 60:
            expression = expression[:57] + '...'
        info = 'Expression = ' + expression + '\n'
        dtype = self.dtype
        if self.expression in self.ds.get_column_names(hidden=True):
            state = "column"
        elif self.expression in self.ds.get_column_names(hidden=True):
            state = "virtual column"
        else:
            state = "expression"
        line = 'Length: {:,} dtype: {} ({})\n'.format(len(self.ds), dtype, state)
        info += line
        info += '-' * (len(line)-1) + '\n'
        info += values
        return info

    def count(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, edges=False, progress=None):
        '''Shortcut for ds.count(expression, ...), see `Dataset.count`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.count(**kwargs)

    def sum(self, axis=None, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Sum elements over given axis.

        If no axis is given, it will sum over all axes.

        For non list elements, this is a shortcut for ds.sum(expression, ...), see `Dataset.sum`.

        >>> list_data = [1, 2, None], None, [], [1, 3, 4, 5]
        >>> df = vaex.from_arrays(some_list=pa.array(list_data))
        >>> df.some_list.sum().item()  # will sum over all axis
        16
        >>> df.some_list.sum(axis=1).tolist()  # sums the list elements
        [3, None, 0, 13]

        :param int axis: Axis over which to determine the unique elements (None will flatten arrays or lists)
        '''
        expression = self
        if axis is None:
            axis = [0]
            dtype = self.dtype
            while dtype.is_list:
                axis.append(axis[-1] + 1)
                dtype = dtype.value_type
        elif not isinstance(axis, list):
            axis = [axis]
            axis = list(set(axis))  # remove repeated elements
        dtype = self.dtype
        if 1 in axis:
            if self.dtype.is_list:
                expression = expression.list_sum()
                if axis:
                    axis.remove(1)
            else:
                raise ValueError(f'axis=1 not supported for dtype={dtype}')
        if axis and axis[0] != 0:
            raise ValueError(f'Only axis 0 or 1 is supported')
        if axis is None or 0 in axis:
            kwargs = dict(locals())
            del kwargs['self']
            del kwargs['axis']
            del kwargs['dtype']
            kwargs['expression'] = expression.expression
            return self.ds.sum(**kwargs)
        else:
            return expression

    def mean(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Shortcut for ds.mean(expression, ...), see `Dataset.mean`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.mean(**kwargs)

    def std(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Shortcut for ds.std(expression, ...), see `Dataset.std`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.std(**kwargs)

    def var(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Shortcut for ds.std(expression, ...), see `Dataset.var`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.var(**kwargs)

    def minmax(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Shortcut for ds.minmax(expression, ...), see `Dataset.minmax`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.minmax(**kwargs)

    def min(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Shortcut for ds.min(expression, ...), see `Dataset.min`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.min(**kwargs)

    def max(self, binby=[], limits=None, shape=default_shape, selection=False, delay=False, progress=None):
        '''Shortcut for ds.max(expression, ...), see `Dataset.max`'''
        kwargs = dict(locals())
        del kwargs['self']
        kwargs['expression'] = self.expression
        return self.ds.max(**kwargs)

    def nop(self):
        """Evaluates expression, and drop the result, usefull for benchmarking, since vaex is usually lazy"""
        return self.ds.nop(self.expression)

    @property
    def transient(self):
        """If this expression is not transient (e.g. on disk) optimizations can be made"""
        return self.expand().expression not in self.ds.columns

    @property
    def masked(self):
        """Alias to df.is_masked(expression)"""
        return self.ds.is_masked(self.expression)

    def value_counts(self, dropna=False, dropnan=False, dropmissing=False, ascending=False, progress=False, axis=None):
        """Computes counts of unique values.

         WARNING:
          * If the expression/column is not categorical, it will be converted on the fly
          * dropna is False by default, it is True by default in pandas

        :param dropna: when True, it will not report the NA (see :func:`Expression.isna`)
        :param dropnan: when True, it will not report the nans(see :func:`Expression.isnan`)
        :param dropmissing: when True, it will not report the missing values (see :func:`Expression.ismissing`)
        :param ascending: when False (default) it will report the most frequent occuring item first
        :param bool axis: Axis over which to determine the unique elements (None will flatten arrays or lists)
        :returns: Pandas series containing the counts
        """
        from pandas import Series
        if axis is not None:
            raise ValueError('only axis=None is supported')

        data_type = self.data_type()
        data_type_item = self.data_type(axis=-1)

        transient = self.transient or self.ds.filtered or self.ds.is_masked(self.expression)
        if self.is_string() and not transient:
            # string is a special case, only ColumnString are not transient
            ar = self.ds.columns[self.expression]
            if not isinstance(ar, ColumnString):
                transient = True

        counter_type = counter_type_from_dtype(data_type_item, transient)
        counters = [None] * self.ds.executor.thread_pool.nthreads
        def map(thread_index, i1, i2, ar):
            if counters[thread_index] is None:
                counters[thread_index] = counter_type()
            if data_type.is_list and axis is None:
                ar = ar.values
            if data_type_item.is_string:
                ar = _to_string_sequence(ar)
            else:
                ar = vaex.array_types.to_numpy(ar)
            if np.ma.isMaskedArray(ar):
                mask = np.ma.getmaskarray(ar)
                counters[thread_index].update(ar, mask)
            else:
                counters[thread_index].update(ar)
            return 0
        def reduce(a, b):
            return a+b
        self.ds.map_reduce(map, reduce, [self.expression], delay=False, progress=progress, name='value_counts', info=True, to_numpy=False)
        counters = [k for k in counters if k is not None]
        counter0 = counters[0]
        for other in counters[1:]:
            counter0.merge(other)
        value_counts = counter0.extract()
        index = np.array(list(value_counts.keys()))
        counts = np.array(list(value_counts.values()))

        order = np.argsort(counts)
        if not ascending:
            order = order[::-1]
        counts = counts[order]
        index = index[order]
        # nan can already be present for dtype=object, remove it
        nan_mask = index != index
        if np.any(nan_mask):
            index = index[~mask]
            counts = index[~mask]
        # nan can already be present for dtype=object, optionally remove it
        none_mask = index == None
        if np.any(none_mask):
            index = index.tolist()
            counts = counts.tolist()
            i = index.index(None)
            if (dropmissing or dropna):
                del index[i]
                del counts[i]
            else:
                index[i] = "missing"
            index = np.array(index)
            counts = np.array(counts)

        if not dropna or not dropnan or not dropmissing:
            index = index.tolist()
            counts = counts.tolist()
            if not (dropnan or dropna) and counter0.nan_count:
                index = [np.nan] + index
                counts = [counter0.nan_count] + counts
            if not (dropmissing or dropna) and counter0.null_count:
                index = ['missing'] + index
                counts = [counter0.null_count] + counts

        return Series(counts, index=index)

    @docsubst
    def unique(self, dropna=False, dropnan=False, dropmissing=False, selection=None, axis=None, array_type='list', delay=False):
        """Returns all unique values.

        :param dropmissing: do not count missing values
        :param dropnan: do not count nan values
        :param dropna: short for any of the above, (see :func:`Expression.isna`)
        :param bool axis: Axis over which to determine the unique elements (None will flatten arrays or lists)
        :param bool array_type: {array_type}
        """
        return self.ds.unique(self.expression, dropna=dropna, dropnan=dropnan, dropmissing=dropmissing, selection=selection, array_type=array_type, axis=axis, delay=delay)

    def nunique(self, dropna=False, dropnan=False, dropmissing=False, selection=None, axis=None, delay=False):
        """Counts number of unique values, i.e. `len(df.x.unique()) == df.x.nunique()`.

        :param dropmissing: do not count missing values
        :param dropnan: do not count nan values
        :param dropna: short for any of the above, (see :func:`Expression.isna`)
        :param bool axis: Axis over which to determine the unique elements (None will flatten arrays or lists)
        """
        return len(self.unique(dropna=dropna, dropnan=dropnan, dropmissing=dropmissing, selection=selection, axis=axis, delay=delay))

    def countna(self):
        """Returns the number of Not Availiable (N/A) values in the expression.
        This includes missing values and np.nan values.
        """
        return self.isna().sum().item()  # so the output is int, not array

    def countnan(self):
        """Returns the number of NaN values in the expression."""
        return self.isnan().sum().item()  # so the output is int, not array

    def countmissing(self):
        """Returns the number of missing values in the expression."""
        return self.ismissing().sum().item()  # so the output is int, not array

    def evaluate(self, i1=None, i2=None, out=None, selection=None, parallel=True, array_type=None):
        return self.ds.evaluate(self, i1, i2, out=out, selection=selection, array_type=array_type, parallel=parallel)

    # TODO: it is not so elegant we need to have a custom version of this
    # it now also misses the docstring, reconsider how the the meta class auto
    # adds this method
    def fillna(self, value, fill_nan=True, fill_masked=True):
        return self.ds.func.fillna(self, value=value, fill_nan=fill_nan, fill_masked=fill_masked)

    def clip(self, lower=None, upper=None):
        return self.ds.func.clip(self, lower, upper)

    def jit_numba(self, verbose=False):
        f = FunctionSerializableNumba.build(self.expression, df=self.ds, verbose=verbose, compile=self.ds.is_local())
        function = self.ds.add_function('_jit', f, unique=True)
        return function(*f.arguments)

    def jit_cuda(self, verbose=False):
        f = FunctionSerializableCuda.build(self.expression, df=self.ds, verbose=verbose, compile=self.ds.is_local())
        function = self.ds.add_function('_jit', f, unique=True)
        return function(*f.arguments)

    def jit_pythran(self, verbose=False):
        import logging
        logger = logging.getLogger('pythran')
        log_level = logger.getEffectiveLevel()
        try:
            if not verbose:
                logger.setLevel(logging.ERROR)
            import pythran
            import imp
            import hashlib
            # self._import_all(module)
            names = []
            funcs = set(expression_namespace.keys())
            expression = self.expression
            if expression in self.ds.virtual_columns:
                expression = self.ds.virtual_columns[self.expression]
            all_vars = self.ds.get_column_names(virtual=True, strings=True, hidden=True) + list(self.ds.variables.keys())
            vaex.expresso.validate_expression(expression, all_vars, funcs, names)
            names = list(set(names))
            types = ", ".join(str(self.ds.data_type(name)) + "[]" for name in names)
            argstring = ", ".join(names)
            code = '''
from numpy import *
#pythran export f({2})
def f({0}):
    return {1}'''.format(argstring, expression, types)
            if verbose:
                print("generated code")
                print(code)
            m = hashlib.md5()
            m.update(code.encode('utf-8'))
            module_name = "pythranized_" + m.hexdigest()
            # print(m.hexdigest())
            module_path = pythran.compile_pythrancode(module_name, code, extra_compile_args=["-DBOOST_SIMD", "-march=native"] + [] if verbose else ["-w"])

            module = imp.load_dynamic(module_name, module_path)
            function_name = "f_" + m.hexdigest()
            function = self.ds.add_function(function_name, module.f, unique=True)

            return Expression(self.ds, "{0}({1})".format(function.name, argstring))
        finally:
                logger.setLevel(log_level)

    def _rename(self, old, new, inplace=False):
        expression = self if inplace else self.copy()
        if old in expression.ast_names:
            for node in expression.ast_names[old]:
                node.id = new
            expression._ast_names[new] = expression._ast_names.pop(old)
        slices = expression._ast_slices
        if old in slices:
            for node in slices[old]:
                node.slice.value.s = new
        expression._expression = None  # resets the cached string representation
        return expression

    def astype(self, data_type):
        if vaex.array_types.is_string_type(data_type) or data_type == str:
            return self.ds.func.astype(self, 'str')
        else:
            return self.ds.func.astype(self, str(data_type))

    def isin(self, values, use_hashmap=True):
        """Lazily tests if each value in the expression is present in values.

        :param values: List/array of values to check
        :param use_hashmap: use a hashmap or not (especially faster when values contains many elements)
        :return: :class:`Expression` with the lazy expression.
        """
        if use_hashmap:
            # easiest way to create a set is using the vaex dataframe
            df_values = vaex.from_arrays(x=values)
            ordered_set = df_values._set(df_values.x)
            var = self.df.add_variable('var_isin_ordered_set', ordered_set, unique=True)
            return self.df['isin_set(%s, %s)' % (self, var)]
        else:
            if self.is_string():
                values = vaex.column._to_string_sequence(values)
            else:
                values = np.array(values, dtype=self.dtype)
            var = self.df.add_variable('isin_values', values, unique=True)
            return self.df['isin(%s, %s)' % (self, var)]

    def apply(self, f, vectorize=False, multiprocessing=True):
        """Apply a function along all values of an Expression.

        Shorthand for ``df.apply(f, arguments=[expression])``, see :meth:`DataFrame.apply`

        Example:

        >>> df = vaex.example()
        >>> df.x
        Expression = x
        Length: 330,000 dtype: float64 (column)
        ---------------------------------------
             0  -0.777471
             1    3.77427
             2    1.37576
             3   -7.06738
             4   0.243441

        >>> def func(x):
        ...     return x**2

        >>> df.x.apply(func)
        Expression = lambda_function(x)
        Length: 330,000 dtype: float64 (expression)
        -------------------------------------------
             0   0.604461
             1    14.2451
             2    1.89272
             3    49.9478
             4  0.0592637


        :param f: A function to be applied on the Expression values
        :param vectorize: Call f with arrays instead of a scalars (for better performance).
        :param bool multiprocessing: Use multiple processes to avoid the GIL (Global interpreter lock).
        :returns: A function that is lazily evaluated when called.
        """
        return self.ds.apply(f, [self.expression], vectorize=vectorize, multiprocessing=multiprocessing)

    def dropmissing(self):
        # TODO: df.dropna does not support inplace
        # df = self.df if inplace else self.df.copy()
        df = self.ds
        df = df.dropmissing(column_names=[self.expression])
        return df._expr(self.expression)

    def dropnan(self):
        # TODO: df.dropna does not support inplace
        # df = self.df if inplace else self.df.copy()
        df = self.ds
        df = df.dropnan(column_names=[self.expression])
        return df._expr(self.expression)

    def dropna(self):
        # TODO: df.dropna does not support inplace
        # df = self.df if inplace else self.df.copy()
        df = self.ds
        df = df.dropna(column_names=[self.expression])
        return df._expr(self.expression)

    def map(self, mapper, nan_value=None, missing_value=None, default_value=None, allow_missing=False, axis=None):
        """Map values of an expression or in memory column according to an input
        dictionary or a custom callable function.

        Example:

        >>> import vaex
        >>> df = vaex.from_arrays(color=['red', 'red', 'blue', 'red', 'green'])
        >>> mapper = {'red': 1, 'blue': 2, 'green': 3}
        >>> df['color_mapped'] = df.color.map(mapper)
        >>> df
        #  color      color_mapped
        0  red                   1
        1  red                   1
        2  blue                  2
        3  red                   1
        4  green                 3
        >>> import numpy as np
        >>> df = vaex.from_arrays(type=[0, 1, 2, 2, 2, np.nan])
        >>> df['role'] = df['type'].map({0: 'admin', 1: 'maintainer', 2: 'user', np.nan: 'unknown'})
        >>> df
        #    type  role
        0       0  admin
        1       1  maintainer
        2       2  user
        3       2  user
        4       2  user
        5     nan  unknown
        >>> import vaex
        >>> import numpy as np
        >>> df = vaex.from_arrays(type=[0, 1, 2, 2, 2, 4])
        >>> df['role'] = df['type'].map({0: 'admin', 1: 'maintainer', 2: 'user'}, default_value='unknown')
        >>> df
        #    type  role
        0       0  admin
        1       1  maintainer
        2       2  user
        3       2  user
        4       2  user
        5       4  unknown
        :param mapper: dict like object used to map the values from keys to values
        :param nan_value: value to be used when a nan is present (and not in the mapper)
        :param missing_value: value to use used when there is a missing value
        :param default_value: value to be used when a value is not in the mapper (like dict.get(key, default))
        :param allow_missing: used to signal that values in the mapper should map to a masked array with missing values,
            assumed True when default_value is not None.
        :param bool axis: Axis over which to determine the unique elements (None will flatten arrays or lists)
        :return: A vaex expression
        :rtype: vaex.expression.Expression
        """
        assert isinstance(mapper, collectionsAbc.Mapping), "mapper should be a dict like object"
        if axis is not None:
            raise ValueError('only axis=None is supported')

        df = self.ds
        mapper_keys = list(mapper.keys())
        try:
            mapper_nan_key_mask = np.isnan(mapper_keys)
        except TypeError:
            # case where we have mixed strings/nan etc
            def try_nan(x):
                try:
                    return np.isnan(x)
                except:
                    return False
            mapper_nan_key_mask = np.array([try_nan(k) for k in mapper_keys])
        mapper_has_nan = mapper_nan_key_mask.sum() > 0
        if mapper_nan_key_mask.sum() > 1:
            raise ValueError('Insanity, you provided multiple nan values as keys for your dict')
        if mapper_has_nan:
            for key, value in mapper.items():
                if key != key:
                    nan_value = value
        for key, value in mapper.items():
            if key is None:
                missing_value = value

        if axis is not None:
            raise ValueError('only axis=None is supported')
        # we map the keys to a ordinal values [0, N-1] using the set
        key_set = df._set(self.expression, flatten=axis is None)
        found_keys = key_set.keys()

        # we want all possible values to be converted
        # so mapper's key should be a superset of the keys found
        use_masked_array = False
        if default_value is not None:
            allow_missing = True
        if allow_missing:
            use_masked_array = True
        if not set(mapper_keys).issuperset(found_keys):
            missing = set(found_keys).difference(mapper_keys)
            missing0 = list(missing)[0]
            only_has_nan = missing0 != missing0 and len(missing) == 1
            if allow_missing:
                if default_value is not None:
                    value0 = list(mapper.values())[0]
                    assert np.issubdtype(type(default_value), np.array(value0).dtype), "default value has to be of similar type"
            else:
                if only_has_nan:
                    pass  # we're good, the hash mapper deals with nan
                else:
                    raise ValueError('Missing %i values in mapper: %s' % (len(missing), missing))

        # and these are the corresponding choices
        # note that here we map 'planned' unknown values to the default values
        # and later on in _choose, we map values not even seen in the dataframe
        # to the default_value
        dtype = self.data_type(self.expression)
        dtype_item = self.data_type(self.expression, axis=-1)
        if dtype_item.is_float:
            print(nan_value)
            values  = [np.nan, None] + [key for key in mapper if key == key and key is not None]
            choices = [default_value, nan_value, missing_value] + [mapper[key] for key in mapper if key == key and key is not None]
        else:
            values  = [None] + [key for key in mapper if key is not None]
            choices = [default_value, missing_value] + [mapper[key] for key in mapper if key is not None]
        values = pa.array(values)
        choices = pa.array(choices)
        from .hash import ordered_set_type_from_dtype
        ordered_set_type = ordered_set_type_from_dtype(dtype_item)
        ordered_set = ordered_set_type()
        if vaex.array_types.is_string_type(dtype_item):
            values = _to_string_sequence(values)
        else:
            values = vaex.array_types.to_numpy(values)
        if np.ma.isMaskedArray(values):
            mask = np.ma.getmaskarray(values)
            ordered_set.update(values.data, mask)
        else:
            ordered_set.update(values)

        key_set_name = df.add_variable('map_key_set', ordered_set, unique=True)
        choices_name = df.add_variable('map_choices', choices, unique=True)
        if allow_missing:
            expr = '_map({}, {}, {}, use_missing={!r}, axis={!r})'.format(self, key_set_name, choices_name, use_masked_array, axis)
        else:
            expr = '_map({}, {}, {}, axis={!r})'.format(self, key_set_name, choices_name, axis)
        return Expression(df, expr)

    @property
    def is_masked(self):
        return self.ds.is_masked(self.expression)

    def is_string(self):
        return self.df.is_string(self.expression)


class FunctionSerializable(object):
    pass


@vaex.serialize.register
class FunctionSerializablePickle(FunctionSerializable):
    def __init__(self, f=None, multiprocessing=False):
        self.f = f
        self.multiprocessing = multiprocessing

    def pickle(self, function):
        return pickle.dumps(function)

    def unpickle(self, data):
        return pickle.loads(data)

    def __getstate__(self):
        return self.state_get()

    def __setstate__(self, state):
        self.state_set(state)

    def state_get(self):
        data = self.pickle(self.f)
        if vaex.utils.PY2:
            pickled = base64.encodestring(data)
        else:
            pickled = base64.encodebytes(data).decode('ascii')
        return dict(pickled=pickled)

    @classmethod
    def state_from(cls, state, trusted=True):
        obj = cls()
        obj.state_set(state, trusted=trusted)
        return obj

    def state_set(self, state, trusted=True):
        data = state['pickled']
        if vaex.utils.PY2:
            data = base64.decodestring(data)
        else:
            data = base64.decodebytes(data.encode('ascii'))
        if trusted is False:
            raise ValueError("Will not unpickle data when source is not trusted")
        self.f = self.unpickle(data)

    def __call__(self, *args, **kwargs):
        '''Forward the call to the real function'''
        import vaex.multiprocessing
        return vaex.multiprocessing.apply(self._apply, args, kwargs, self.multiprocessing)

    def _apply(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class FunctionSerializableJit(FunctionSerializable):
    def __init__(self, expression, arguments, argument_dtypes, return_dtype, verbose=False, compile=True):
        self.expression = expression
        self.arguments = arguments
        self.argument_dtypes = argument_dtypes
        self.return_dtype = return_dtype
        self.verbose = verbose
        if compile:
            self.f = self.compile()
        else:
            def placeholder(*args, **kwargs):
                raise Exception('You chose not to compile this function (locally), but did invoke it')
            self.f = placeholder

    def state_get(self):
        return dict(expression=self.expression,
                    arguments=self.arguments,
                    argument_dtypes=list(map(lambda dtype: str(dtype.numpy), self.argument_dtypes)),
                    return_dtype=str(self.return_dtype),
                    verbose=self.verbose)

    @classmethod
    def state_from(cls, state, trusted=True):
        return cls(expression=state['expression'],
                   arguments=state['arguments'],
                   argument_dtypes=list(map(lambda s: DataType(np.dtype(s)), state['argument_dtypes'])),
                   return_dtype=DataType(np.dtype(state['return_dtype'])),
                   verbose=state['verbose'])

    @classmethod
    def build(cls, expression, df=None, verbose=False, compile=True):
        df = df or expression.df
        # if it's a virtual column, we probably want to optimize that
        # TODO: fully extract the virtual columns, i.e. depending ones?
        expression = str(expression)

        if expression in df.virtual_columns:
            expression = df.virtual_columns[expression]

        # function validation, and finding variable names
        all_vars = df.get_column_names(hidden=True) + list(df.variables.keys())
        funcs = set(list(expression_namespace.keys()) + list(df.functions.keys()))
        names = []
        vaex.expresso.validate_expression(expression, all_vars, funcs, names)
        # TODO: can we do the above using the Expressio API?s

        arguments = list(set(names))
        argument_dtypes = [df.data_type(argument, array_type='numpy') for argument in arguments]
        return_dtype = df[expression].dtype
        return cls(str(expression), arguments, argument_dtypes, return_dtype, verbose, compile=compile)

    def __call__(self, *args, **kwargs):
        '''Forward the call to the numba function'''
        return self.f(*args, **kwargs)


@vaex.serialize.register
class FunctionSerializableNumba(FunctionSerializableJit):
    def compile(self):
        import numba
        argstring = ", ".join(self.arguments)
        code = '''
from numpy import *
def f({0}):
    return {1}'''.format(argstring, self.expression)
        if self.verbose:
            print('Generated code:\n' + code)
        scope = {}
        exec(code, scope)
        f = scope['f']

        # numba part
        def get_type(name):
            if name == "bool":
                name = "bool_"
            return getattr(numba, name)

        argument_dtypes_numba = [get_type(argument_dtype.numpy.name) for argument_dtype in self.argument_dtypes]
        return_dtype_numba = get_type(self.return_dtype.numpy.name)
        vectorizer = numba.vectorize([return_dtype_numba(*argument_dtypes_numba)])
        return vectorizer(f)


@vaex.serialize.register
class FunctionSerializableCuda(FunctionSerializableJit):
    def compile(self):
        import cupy
        # code generation
        argstring = ", ".join(self.arguments)
        code = '''
from cupy import *
import cupy
@fuse()
def f({0}):
    return {1}
'''.format(argstring, self.expression)#, ";".join(conversions))
        if self.verbose:
            print("generated code")
            print(code)
        scope = dict()#cupy=cupy)
        exec(code, scope)
        func = scope['f']
        def wrapper(*args):
            args = [vaex.array_types.to_numpy(k) for k in args]
            args = [vaex.utils.to_native_array(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
            args = [cupy.asarray(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
            return cupy.asnumpy(func(*args))
        return wrapper


# TODO: this is not the right abstraction, since this won't allow a
# numba version for the function
@vaex.serialize.register
class FunctionToScalar(FunctionSerializablePickle):
    def __call__(self, *args, **kwargs):
        import vaex.multiprocessing
        return vaex.multiprocessing.apply(self._apply, args, kwargs, self.multiprocessing)

    def _apply(self, *args, **kwargs):
        length = len(args[0])
        result = []
        def fix_type(v):
            # TODO: only when column is str type?
            if isinstance(v, np.str_):
                return str(v)
            if isinstance(v, np.bytes_):
                return v.decode('utf8')
            else:
                return v
        args = [vaex.array_types.tolist(k) for k in args]
        for i in range(length):
            scalar_result = self.f(*[fix_type(k[i]) for k in args], **{key: value[i] for key, value in kwargs.items()})
            result.append(scalar_result)
        result = np.array(result)
        return result


class Function(object):

    def __init__(self, dataset, name, f):
        self.dataset = dataset
        self.name = name

        if not vaex.serialize.can_serialize(f): # if not serializable, assume we can use pickle
            f = FunctionSerializablePickle(f)
        self.f = f

    def __call__(self, *args, **kwargs):
        arg_string = ", ".join([str(k) for k in args] + ['{}={:r}'.format(name, value) for name, value in kwargs.items()])
        expression = "{}({})".format(self.name, arg_string)
        return Expression(self.dataset, expression)


class FunctionBuiltin(object):

    def __init__(self, dataset, name, **kwargs):
        self.dataset = dataset
        self.name = name
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        kwargs = dict(kwargs, **self.kwargs)
        arg_string = ", ".join([str(k) for k in args] + ['{}={:r}'.format(name, value) for name, value in kwargs.items()])
        expression = "{}({})".format(self.name, arg_string)
        return Expression(self.dataset, expression)

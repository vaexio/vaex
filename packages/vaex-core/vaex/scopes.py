from __future__ import division, print_function
import logging
import numpy as np
import pyarrow as pa
import vaex.array_types
import vaex.arrow.numpy_dispatch


from .utils import (_ensure_strings_from_expressions,
    _ensure_string_from_expression,
    _ensure_list,
    _is_limit,
    _isnumber,
    _issequence,
    _is_string,
    _parse_reduction,
    _parse_n,
    _normalize_selection_name,
    _normalize,
    _parse_f,
    _expand,
    _expand_shape,
    _expand_limits,
    as_flat_float,
    as_flat_array,
    _split_and_combine_mask)
from .expression import expression_namespace
from vaex.arrow.numpy_dispatch import wrap, unwrap
import vaex.expression

logger = logging.getLogger('vaex.scopes')


class ScopeBase(object):
    def get(self, attr, default=None):  # otherwise pdb crashes during pytest
        if attr == "__tracebackhide__":
            return False
        return default


def auto_encode(df, expression, values):
    return df._auto_encode_data(expression, values)


class UnitScope(ScopeBase):
    def __init__(self, df, value=None):
        self.df = df
        self.value = value

    def __getitem__(self, variable):
        import astropy.units
        if variable in self.df.units:
            unit = self.df.units[variable]
            return (self.value * unit) if self.value is not None else unit
        elif variable in self.df.virtual_columns:
            return eval(self.df.virtual_columns[variable], expression_namespace, self)
        elif variable in self.df.variables:
            return astropy.units.dimensionless_unscaled  # TODO units for variables?
        else:
            raise KeyError("unkown variable %s" % variable)


class _BlockScope(ScopeBase):
    def __init__(self, df, i1, i2, mask=None, **variables):
        """

        :param DataFrameLocal DataFrame: the *local*  DataFrame
        :param i1: start index
        :param i2: end index
        :param values:
        :return:
        """
        self.df = df
        self.i1 = int(i1)
        self.i2 = int(i2)
        self.variables = variables
        self.values = dict(self.variables)
        self.buffers = {}
        self.mask = mask if mask is not None else None

    def move(self, i1, i2):
        length_new = i2 - i1
        length_old = self.i2 - self.i1
        if length_new > length_old:  # old buffers are too small, discard them
            self.buffers = {}
        else:
            for name in list(self.buffers.keys()):
                self.buffers[name] = self.buffers[name][:length_new]
        self.i1 = int(i1)
        self.i2 = int(i2)
        self.values = dict(self.variables)

    def __contains__(self, name):  # otherwise pdb crashes during pytest
        return name in self.buffers  # not sure this should also include varibles, columns and virtual columns

    def _ensure_buffer(self, column):
        if column not in self.buffers:
            logger.debug("creating column for: %s", column)
            self.buffers[column] = np.zeros(self.i2 - self.i1)

    def evaluate(self, expression, out=None):
        if isinstance(expression, vaex.expression.Expression):
            expression = expression.expression
        try:
            # logger.debug("try avoid evaluating: %s", expression)
            result = self[expression]
        except KeyError:
            # logger.debug("no luck, eval: %s", expression)
            # result = ne.evaluate(expression, local_dict=self, out=out)
            # logger.debug("in eval")
            # eval("def f(")
            result = eval(expression, expression_namespace, self)
            result = auto_encode(self.df, expression, result)
            self.values[expression] = wrap(result)
            # if out is not None:
            #   out[:] = result
            #   result = out
            # logger.debug("out eval")
        # logger.debug("done with eval of %s", expression)
        result = unwrap(result)
        return result

    def __getitem__(self, variable):
        # logger.debug("get " + variable)
        # return self.df.columns[variable][self.i1:self.i2]
        if variable == 'df':
            return self  # to support df['no!identifier']
        try:
            if variable in self.values:
                return self.values[variable]
            elif variable in self.df.columns:
                offset = self.df._index_start
                # if self.df._needs_copy(variable):
                    # self._ensure_buffer(variable)
                    # self.values[variable] = self.buffers[variable] = self.df.columns[variable][self.i1:self.i2].astype(np.float64)
                    # Previously we casted anything to .astype(np.float64), this led to rounding off of int64, when exporting
                    # self.values[variable] = self.df.columns[variable][offset+self.i1:offset+self.i2][:]
                # else:
                values = self.df.columns[variable][offset+self.i1:offset+self.i2]
                if self.mask is not None:
                    # TODO: we may want to put this in array_types
                    if isinstance(values, (pa.Array, pa.ChunkedArray)):
                        values = values.filter(vaex.array_types.to_arrow(self.mask))
                    else:
                        values = values[self.mask]
                values = auto_encode(self.df, variable, values)
                self.values[variable] = wrap(values)
            elif variable in list(self.df.virtual_columns.keys()):
                expression = self.df.virtual_columns[variable]
                if isinstance(expression, dict):
                    function = expression['function']
                    arguments = [self.evaluate(k) for k in expression['arguments']]
                    values = function(*arguments)
                else:
                    # self._ensure_buffer(variable)
                    values = self.evaluate(expression)
                values = auto_encode(self.df, variable, values)
                self.values[variable] = wrap(values)
                # self.values[variable] = self.buffers[variable]
            elif variable in self.df.functions:
                f = self.df.functions[variable].f
                return vaex.arrow.numpy_dispatch.autowrapper(f)
            elif variable in expression_namespace:
                return expression_namespace[variable]
            if variable not in self.values:
                raise KeyError("Unknown variables or column: %r" % (variable,))

            return self.values[variable]
        except:
            # logger.exception("error in evaluating: %r" % variable)
            raise


class _BlockScopeSelection(ScopeBase):
    def __init__(self, df, i1, i2, selection=None, cache=False, filter_mask=None):
        self.df = df
        self.i1 = i1
        self.i2 = i2
        self.selection = selection
        self.store_in_cache = cache
        self.filter_mask = filter_mask

    def evaluate(self, expression):
        if expression is True:
            expression = "default"
        try:
            expression = _ensure_string_from_expression(expression)
            result = eval(expression, expression_namespace, self)
        except:
            import traceback as tb
            tb.print_stack()
            raise
        result = unwrap(result)
        return result

    def __contains__(self, name):  # otherwise pdb crashes during pytest
        return False

    def __getitem__(self, variable):
        if variable == "__tracebackhide__":  # required for tracebacks
            return False
        if variable == 'df':
            return self  # to support df['no!identifier']
        # logger.debug("getitem for selection: %s", variable)
        try:
            selection = self.selection
            if selection is None and self.df.has_selection(variable):
                selection = self.df.get_selection(variable)
            # logger.debug("selection for %r: %s %r", variable, selection, self.df.selection_histories)
            key = (self.i1, self.i2)
            if selection:
                assert variable in self.df._selection_masks, "%s mask not found" % (variable, )
                cache = self.df._selection_mask_caches[variable]
                # logger.debug("selection cache: %r" % cache)
                full_mask = self.df._selection_masks[variable]
                selection_in_cache, mask = cache.get(key, (None, None))

                # logger.debug("mask for %r is %r", variable, mask)
                if selection_in_cache == selection:
                    if self.filter_mask is not None:
                        return wrap(mask[self.filter_mask])
                    return wrap(mask)
                # logger.debug("was not cached")
                if variable in self.df.variables:
                    return wrap(self.df.variables[variable])
                mask_values = selection.evaluate(self.df, variable, self.i1, self.i2, self.filter_mask)
                    
                # get a view on a subset of the mask
                sub_mask = full_mask.view(self.i1, self.i2)
                sub_mask_array = np.asarray(sub_mask)
                # and update it
                if self.filter_mask is not None:  # if we have a mask, the selection we evaluated is also filtered
                    sub_mask_array[:] = 0
                    sub_mask_array[:][self.filter_mask] = mask_values
                else:
                    sub_mask_array[:] = mask_values
                # logger.debug("put selection in mask with key %r" % (key,))
                if self.store_in_cache:
                    cache[key] = selection, sub_mask_array
                    # cache[key] = selection, mask_values
                if self.filter_mask is not None:
                    return wrap(sub_mask_array[self.filter_mask])
                else:
                    return wrap(sub_mask_array)
                # return mask_values
            else:
                offset = self.df._index_start
                if variable in self.df.columns:
                    values = self.df.columns[variable][offset+self.i1:offset+self.i2]
                    # TODO: we may want to put this in array_types
                    if self.filter_mask is not None:
                        if isinstance(values, (pa.Array, pa.ChunkedArray)):
                            values = values.filter(vaex.array_types.to_arrow(self.filter_mask))
                        else:
                            values = values[self.filter_mask]
                    values = auto_encode(self.df, variable, values)
                    return wrap(values)
                elif variable in self.df.variables:
                    return self.df.variables[variable]
                elif variable in self.df.virtual_columns:
                    expression = self.df.virtual_columns[variable]
                    # self._ensure_buffer(variable)
                    if expression == variable:
                        raise ValueError(f'Recursion protection: virtual column {variable} refers to itself')
                    values = self.evaluate(expression)  # , out=self.buffers[variable])
                    values = auto_encode(self.df, variable, values)
                    return wrap(values)
                elif variable in self.df.functions:
                    f = self.df.functions[variable].f
                    return vaex.arrow.numpy_dispatch.autowrapper(f)
                elif variable in expression_namespace:
                    return wrap(expression_namespace[variable])
                raise KeyError("Unknown variables or column: %r" % (variable,))
        except:
            import traceback as tb
            tb.print_exc()
            logger.exception("error in evaluating: %r" % variable)
            raise


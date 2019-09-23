import logging
import numpy as np

import vaex.expression
import vaex.functions
from .utils import _split_and_combine_mask, as_flat_float

logger = logging.getLogger('vaex.selections')


def _select_replace(maskold, masknew):
    return masknew


def _select_and(maskold, masknew):
    return masknew if maskold is None else maskold & masknew


def _select_or(maskold, masknew):
    return masknew if maskold is None else maskold | masknew


def _select_xor(maskold, masknew):
    return masknew if maskold is None else maskold ^ masknew


def _select_subtract(maskold, masknew):
    return ~masknew if maskold is None else (maskold) & ~masknew


_select_functions = {"replace": _select_replace,
                     "and": _select_and,
                     "or": _select_or,
                     "xor": _select_xor,
                     "subtract": _select_subtract
                     }

class Selection(object):
    def __init__(self, previous_selection, mode):
        # we don't care about the previous selection if we simply replace the current selection
        self.previous_selection = previous_selection if mode != "replace" else None
        self.mode = mode

    def execute(self, datexecutor, execute_fully=False):
        if execute_fully and self.previous_selection:
            self.previous_selection.execute(executor=executor, execute_fully=execute_fully)

    def _depending_columns(self, ds):
        '''Find all columns that this selection depends on for df ds'''
        depending = set()
        for expression in self.expressions:
            expression = ds._expr(expression)  # make sure it is an expression
            depending |= expression.variables()
        if self.previous_selection:
            depending |= self.previous_selection._depending_columns(ds)
        return depending


class SelectionDropNa(Selection):
    def __init__(self, drop_nan, drop_masked, column_names, previous_selection, mode):
        super(SelectionDropNa, self).__init__(previous_selection, mode)
        self.drop_nan = drop_nan
        self.drop_masked = drop_masked
        self.column_names = column_names
        self.expressions = self.column_names

    def to_dict(self):
        previous = None
        if self.previous_selection:
            previous = self.previous_selection.to_dict()
        return dict(type="dropna", drop_nan=self.drop_nan, drop_masked=self.drop_masked, column_names=self.column_names,
                    mode=self.mode, previous_selection=previous)

    def _rename(self, df, old, new):
        pass  # TODO: do we need to rename the column_names?

    def evaluate(self, df, name, i1, i2):
        if self.previous_selection:
            previous_mask = df.evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
        else:
            previous_mask = None
        mask = np.ones(i2 - i1, dtype=np.bool)
        for name in self.column_names:
            data = df._evaluate(name, i1, i2)
            if self.drop_nan and self.drop_masked:
                mask &= ~vaex.functions.isna(data)
            elif self.drop_nan:
                mask &= ~vaex.functions.isnan(data)
            elif self.drop_masked:
                mask &= ~vaex.functions.ismissing(data)
        if previous_mask is None:
            logger.debug("setting mask")
        else:
            logger.debug("combining previous mask with current mask using op %r", self.mode)
            mode_function = _select_functions[self.mode]
            mask = mode_function(previous_mask, mask)
        return mask


def _rename_expression_string(df, e, old, new):
    return vaex.expression.Expression(self.df, self.boolean_expression)._rename(old, new).expression

class SelectionExpression(Selection):
    def __init__(self,  boolean_expression, previous_selection, mode):
        super(SelectionExpression, self).__init__(previous_selection, mode)
        self.boolean_expression = str(boolean_expression)
        self.expressions = [self.boolean_expression]

    def _rename(self, df, old, new):
        boolean_expression = vaex.expression.Expression(df, self.boolean_expression)._rename(old, new).expression
        previous_selection = None
        if self.previous_selection:
            previous_selection = self.previous_selection._rename(df, old, new)
        return SelectionExpression(boolean_expression, previous_selection, self.mode)

    def to_dict(self):
        previous = None
        if self.previous_selection:
            previous = self.previous_selection.to_dict()
        return dict(type="expression", boolean_expression=str(self.boolean_expression), mode=self.mode, previous_selection=previous)

    def evaluate(self, df, name, i1, i2):
        if self.previous_selection:
            previous_mask = df._evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
        else:
            previous_mask = None
        current_mask = df._evaluate_selection_mask(self.boolean_expression, i1, i2).astype(np.bool)
        if previous_mask is None:
            logger.debug("setting mask")
            mask = current_mask
        else:
            logger.debug("combining previous mask with current mask using op %r", self.mode)
            mode_function = _select_functions[self.mode]
            mask = mode_function(previous_mask, current_mask)
        return mask


class SelectionInvert(Selection):
    def __init__(self, previous_selection):
        super(SelectionInvert, self).__init__(previous_selection, "")
        self.expressions = []

    def to_dict(self):
        previous = None
        if self.previous_selection:
            previous = self.previous_selection.to_dict()
        return dict(type="invert", previous_selection=previous)

    def evaluate(self, df, name, i1, i2):
        previous_mask = df._evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
        return ~previous_mask


class SelectionLasso(Selection):
    def __init__(self, boolean_expression_x, boolean_expression_y, xseq, yseq, previous_selection, mode):
        super(SelectionLasso, self).__init__(previous_selection, mode)
        self.boolean_expression_x = boolean_expression_x
        self.boolean_expression_y = boolean_expression_y
        self.xseq = xseq
        self.yseq = yseq
        self.expressions = [boolean_expression_x, boolean_expression_y]

    def evaluate(self, df, name, i1, i2):
        if self.previous_selection:
            previous_mask = df._evaluate_selection_mask(name, i1, i2, selection=self.previous_selection)
        else:
            previous_mask = None
        current_mask = np.zeros(i2 - i1, dtype=np.bool)
        x, y = np.array(self.xseq, dtype=np.float64), np.array(self.yseq, dtype=np.float64)
        meanx = x.mean()
        meany = y.mean()
        radius = np.sqrt((meanx - x)**2 + (meany - y)**2).max()
        blockx = df._evaluate(self.boolean_expression_x, i1=i1, i2=i2)
        blocky = df._evaluate(self.boolean_expression_y, i1=i1, i2=i2)
        (blockx, blocky), excluding_mask = _split_and_combine_mask([blockx, blocky])
        blockx = as_flat_float(blockx)
        blocky = as_flat_float(blocky)
        vaex.vaexfast.pnpoly(x, y, blockx, blocky, current_mask, meanx, meany, radius)
        if previous_mask is None:
            logger.debug("setting mask")
            mask = current_mask
        else:
            logger.debug("combining previous mask with current mask using op %r", self.mode)
            mode_function = _select_functions[self.mode]
            mask = mode_function(previous_mask, current_mask)
        if excluding_mask is not None:
            mask = mask & (~excluding_mask)
        return mask

    def to_dict(self):
        previous = None
        if self.previous_selection:
            previous = self.previous_selection.to_dict()
        return dict(type="lasso",
                    boolean_expression_x=str(self.boolean_expression_x),
                    boolean_expression_y=str(self.boolean_expression_y),
                    xseq=vaex.utils.make_list(self.xseq),
                    yseq=vaex.utils.make_list(self.yseq),
                    mode=self.mode,
                    previous_selection=previous)

def selection_from_dict(values):
    kwargs = dict(values)
    del kwargs["type"]
    if values["type"] == "lasso":
        kwargs["previous_selection"] = selection_from_dict(values["previous_selection"]) if values["previous_selection"] else None
        return SelectionLasso(**kwargs)
    elif values["type"] == "expression":
        kwargs["previous_selection"] = selection_from_dict(values["previous_selection"]) if values["previous_selection"] else None
        return SelectionExpression(**kwargs)
    elif values["type"] == "invert":
        kwargs["previous_selection"] = selection_from_dict(values["previous_selection"]) if values["previous_selection"] else None
        return SelectionInvert(**kwargs)
    elif values["type"] == "dropna":
        kwargs["previous_selection"] = selection_from_dict(values["previous_selection"]) if values["previous_selection"] else None
        return SelectionDropNa(**kwargs)
    else:
        raise ValueError("unknown type: %r, in dict: %r" % (values["type"], values))

import numpy as np
import vaex


class GrouperBinby:
    def __init__(self, expression, df=None):
        self.df = df or expression.ds
        self.expression = expression

class GrouperTime(GrouperBinby):
    def __init__(self, expression, freq='W', df=None):
        self.expression = expression
        self.df = df or expression.ds
        self.binby

class Grouper(GrouperBinby):
    def __init__(self, expression, df=None):
        self.df = df or expression.ds
        self.expression = expression
        self.set = self.df._set(self.expression)
        self.set_name = 'test'


class GroupBy(object):
    def __init__(self, df, by):
        self.df = df
        self.by = Grouper(df[str(by)])
        # self._waslist, [self.by, ] = vaex.utils.listify(by)

    def size(self):
        import pandas as pd
        result = self.df.count(binby=self.by, shape=[10000] * len(self.by)).astype(np.int64)
        #values = vaex.utils.unlistify(self._waslist, result)
        values = result
        series = pd.Series(values, index=self.df.category_labels(self.by[0]))
        return series

    def agg(self, actions):
        df = self.df.copy()
        ordered_set = self.by.set
        df.add_variable('myset', ordered_set)

        keys, indices = zip(*ordered_set.extract().items())
        indices = np.array(indices)
        # indices = np.arange(len(indices))[indices]
        keys = np.array(keys)#[indices].tolist()
        keys[indices] = keys.copy()
        keys = keys.tolist()
        binby = '_ordinal_values(%s, myset)' % self.by.expression
        N = len(ordered_set.keys())
        if ordered_set.has_null:
            N += 1
            keys += ['null']
        limits = [-0.5, N-0.5]
        dfg = vaex.from_dict({str(self.by.expression): keys})
        for key, value in actions.items():
            method = getattr(df, value)
            values = method(key, binby=binby, limits=limits, shape=N, delay=True)
            @vaex.delayed
            def done(values, key=key):
                dfg[key] = values
            done(values)
        df.execute()  
        return dfg
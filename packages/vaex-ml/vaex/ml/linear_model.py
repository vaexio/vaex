import numpy as np
import traitlets
import sklearn.linear_model as lin

import vaex
import vaex.serialize
from . import state
from . import generate


def listify(l):
    if hasattr(l, 'tolist'):
        return l.tolist()
    else:
        try:
            return [listify(k) for k in l]
        except TypeError:
            return l


class _LinearBase(state.HasState):
    features = traitlets.List(traitlets.Unicode())
    binned = traitlets.CBool(True)
    shape = traitlets.CInt(64)
    limits = traitlets.List(traitlets.List(traitlets.CFloat(), allow_none=True),
                            allow_none=True).tag(output=True)
    fit_intercept = traitlets.CBool(True)
    coef_ = traitlets.Union(
                            [traitlets.List(traitlets.CFloat()),
                             traitlets.List(traitlets.List(traitlets.CFloat()))]
                            ).tag(output=True)
    # intercept_ = traitlets.List(traitlets.CFloat).tag(output=True)
    intercept_ = traitlets.Union([traitlets.CFloat(),
                                 traitlets.List(traitlets.CFloat())]).tag(output=True)
    _sk_params = traitlets.Any()
    prediction_name = traitlets.Unicode(default_value='linear_prediction')

    def transform(self, dataset):
        ds = dataset.copy()
        expression = self.coef_[0] * ds[self.features[0]]
        for coef, feature in zip(self.coef_, self.features[1:]):
            expression = expression + coef * ds[feature]
        expression = self.intercept_ + expression
        ds.add_virtual_column(self.prediction_name, expression, unique=False)
        return ds

    def fit(self, dataset, y_expression, progress=False):
        assert len(set(self.features)) == len(self.features), "duplicate features"
        if not self.binned:
            X = np.array(dataset[self.features])
            y = dataset.evaluate(y_expression)
            m = self._make_model()
            m.fit(X, y)
            self.coef_ = m.coef_.tolist()
            self.intercept_ = m.intercept_.tolist()
        else:
            limits = self.limits
            if limits == []:
                limits = None
            binby = self.features + [y_expression]
            limits, shapes = dataset.limits(binby, limits, shape=self.shape)
            self.limits = listify(limits)
            counts = dataset.count(binby=binby, limits=limits, shape=shapes)
            mask = counts > 0

            def coordinates(expression, limits, shape):
                if dataset.is_category(expression):
                    return np.arange(dataset.category_count(expression))
                else:
                    return dataset.bin_centers(expression, limits, shape)
            centers = [coordinates(expression, l, shape) for expression, l, shape
                       in zip(binby, self.limits, shapes)]
            # l = ds.bin_centers('y', limits[1], shape)
            centers = np.meshgrid(*centers, indexing='ij')
            centers = [c[mask] for c in centers]
            # m = lin.LinearRegression(fit_intercept=self.fit_intercept)
            m = self._make_model()
            X = np.array(centers[:-1]).reshape(-1, len(self.features))
            y = centers[-1].reshape(-1)
            weights = counts[mask]
            m.fit(X, y, sample_weight=weights)
            self.coef_ = m.coef_.tolist()
            self.intercept_ = m.intercept_.tolist()
        # self._sk_params = m.get_params()
        self.last_model = m

    def predict(self, dataset):
        X = np.array(dataset[self.features])
        return self.last_model.predict(X)


@generate.register
@vaex.serialize.register
class LogisticRegression(_LinearBase):
    prediction_name = traitlets.Unicode(default_value='logit_prediction')

    def _make_model(self):
        return lin.LogisticRegression(fit_intercept=self.fit_intercept)


@generate.register
@vaex.serialize.register
class LinearRegression(_LinearBase):
    prediction_name = traitlets.Unicode(default_value='linear_prediction')

    def _make_model(self):
        return lin.LinearRegression(fit_intercept=self.fit_intercept)


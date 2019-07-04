import vaex.ml
import numpy as np
import vaex.jupyter
import os
import traitlets

dev = os.environ.get('VOILA', None) == None

ident = lambda x: x

def numpy_to_json(ar):
    return ar.tolist() if ar is not None else None
def json_to_numpy(obj):
    return np.array(obj)
serialize_numpy = dict(serialize=numpy_to_json, deserialize=json_to_numpy)

class VizBaseState(vaex.ml.state.HasState):
    shape = traitlets.CInt(64)

    def __init__(self, ds, **kwargs):
        super(VizBaseState, self).__init__(**kwargs)
        self.ds = ds
        self.signal_slice = vaex.events.Signal()
        self.signal_regrid = vaex.events.Signal()

    def state_get(self):
        state = {}
        for name in self.trait_names():
            serializer = self.trait_metadata(name, 'serialize', ident)
            value = serializer(getattr(self, name))
            state[name] = value
        return state

    def state_set(self, state):
        for name in self.trait_names():
            if name in state:
                deserializer = self.trait_metadata(name, 'deserialize', ident)
                value = deserializer(state[name])
                setattr(self, name, value)

    def _calculate_limits(self, attr='x', expression='x_expression'):
        expression = getattr(self, expression)
        categorical = self.ds.iscategory(expression)
        if categorical:
            N = self.ds.category_count(expression)
            min, max = -0.5, N-0.5
            centers = np.arange(N)
            setattr(self, attr + '_shape', N)
        else:
            min, max = self.ds.minmax(expression)
            centers = self.ds.bin_centers(expression, [min, max], shape=getattr(self, attr + '_shape') or self.shape)
        setattr(self, attr + '_min', min)
        setattr(self, attr + '_max', max)
        setattr(self, attr + '_centers', centers)

    def _calculate_centers(self, attr='x', expression='x_expression'):
        expression = getattr(self, expression)
        categorical = self.ds.iscategory(expression)
        min, max = getattr(self, attr + '_min'), getattr(self, attr + '_max')
        if min is None or max is None:
            return # special condition that can occur during testing, since debounced does not work
        if categorical:
            N = self.ds.category_count(expression)
            centers = np.arange(N)
            setattr(self, attr + '_shape', N)
        else:
            # print(expression, [min, max], getattr(self, attr + '_shape') or self.shape)
            centers = self.ds.bin_centers(expression, [min, max], shape=getattr(self, attr + '_shape') or self.shape)
        setattr(self, attr + '_centers', centers)
                                                                      
class VizHistogramState(VizBaseState):
    x_expression = traitlets.Unicode()
    x_slice = traitlets.CInt(None, allow_none=True)
    type = traitlets.CaselessStrEnum(['count', 'min', 'max', 'mean'], default_value='count')
    aux = traitlets.Unicode(None, allow_none=True)
    groupby = traitlets.Unicode(None, allow_none=True)
    groupby_normalize = traitlets.Bool(False, allow_none=True)
    x_min = traitlets.CFloat(None, allow_none=True)
    x_max = traitlets.CFloat(None, allow_none=True)
    grid = traitlets.Any().tag(**serialize_numpy)
    grid_sliced = traitlets.Any().tag(**serialize_numpy)
    x_centers = traitlets.Any().tag(**serialize_numpy)
    x_shape = traitlets.CInt(None, allow_none=True)
    #centers = traitlets.Any()
    
    def __init__(self, ds, **kwargs):
        super(VizHistogramState, self).__init__(ds, **kwargs)
        self.observe(lambda x: self.signal_slice.emit(self), ['x_slice'])
        self.observe(lambda x: self.calculate_limits(), ['x_expression', 'type', 'aux'])
        # no need for recompute
        # self.observe(lambda x: self.calculate_grid(), ['groupby', 'shape', 'groupby_normalize'])
        # self.observe(lambda x: self.calculate_grid(), ['groupby', 'shape', 'groupby_normalize'])
        
        self.observe(lambda x: self._update_grid(), ['x_min', 'x_max', 'shape'])
        if self.x_min is None and self.x_max is None:
            self.calculate_limits()
        else:
            self._calculate_centers()

    def bin_parameters(self):
        yield self.x_expression, self.x_shape or self.shape, (self.x_min, self.x_max), self.x_slice

    def state_get(self):
        #         return {name: self.trait_metadata('grid', 'serialize', ident)(getattr(self, name) for name in self.trait_names()}
        state = {}
        for name in self.trait_names():
            serializer = self.trait_metadata(name, 'serialize', ident)
            value = serializer(getattr(self, name))
            state[name] = value
        return state

    def state_set(self, state):
        for name in self.trait_names():
            if name in state:
                deserializer = self.trait_metadata(name, 'deserialize', ident)
                value = deserializer(state[name])
                setattr(self, name, value)
                                                                      
    def calculate_limits(self):
        self._calculate_limits('x', 'x_expression')
        self.signal_regrid.emit(None) # TODO this is also called in the ctor, unnec work
    
    def limits_changed(self, change):
        self.signal_regrid.emit(None) # TODO this is also called in the ctor, unnec work

    @vaex.jupyter.debounced()
    def _update_grid(self):
        self._calculate_centers()
        self.signal_regrid.emit(None)
            
class VizBase2dState(VizBaseState):
    x_expression = traitlets.Unicode()
    y_expression = traitlets.Unicode()
    x_slice = traitlets.CInt(None, allow_none=True)
    y_slice = traitlets.CInt(None, allow_none=True)
    type = traitlets.CaselessStrEnum(['count', 'min', 'max', 'mean'], default_value='count')
    aux = traitlets.Unicode(None, allow_none=True)
    groupby = traitlets.Unicode(None, allow_none=True)
    x_shape = traitlets.CInt(None, allow_none=True)
    y_shape = traitlets.CInt(None, allow_none=True)

    x_min = traitlets.CFloat()
    x_max = traitlets.CFloat()
    y_min = traitlets.CFloat()
    y_max = traitlets.CFloat()
    
    def __init__(self, ds, **kwargs):
        super(VizBase2dState, self).__init__(ds, **kwargs)
        self.observe(lambda x: self.calculate_limits(), ['x_expression', 'y_expression', 'type', 'aux'])
        self.observe(lambda x: self.signal_slice.emit(self), ['x_slice', 'y_slice'])
        # no need for recompute
        #self.observe(lambda x: self.calculate_grid(), ['groupby', 'shape', 'groupby_normalize'])
        self.observe(self.limits_changed, ['x_min', 'x_max', 'y_min', 'y_max'])
        self.calculate_limits()

    def bin_parameters(self):
        yield self.x_expression, self.x_shape or self.shape, (self.x_min, self.x_max), self.x_slice
        yield self.y_expression, self.y_shape or self.shape, (self.y_min, self.y_max), self.y_slice

    def calculate_limits(self):
        self._calculate_limits('x', 'x_expression')
        self._calculate_limits('y', 'y_expression')
        self.signal_regrid.emit(self)
    
    def limits_changed(self, change):
        self.signal_regrid.emit(self)



class VizHeatmapState(VizBase2dState):
    groupby_normalize = traitlets.Bool(False, allow_none=True)
    grid = traitlets.Any().tag(**serialize_numpy)
    grid_sliced = traitlets.Any().tag(**serialize_numpy)
    x_centers = traitlets.Any().tag(**serialize_numpy)
    #centers = traitlets.Any()
    
    def __init__(self, ds, **kwargs):
        self.ds = ds
        super(VizHeatmapState, self).__init__(ds, **kwargs)

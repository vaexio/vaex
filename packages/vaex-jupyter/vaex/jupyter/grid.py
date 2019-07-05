import vaex
import numpy as np
import vaex.jupyter


class Grid:
    def __init__(self, ds, states):
        self.ds = ds
        self.states = []
        self._callbacks_regrid = []
        self._callbacks_slice = []
        for state in states:
            self.state_add(state, regrid=False)
        self.regrid()
    
    def state_remove(self, state, regrid=True):
        index = self.states.index(state)
        del self.states[index]
        del self._callbacks_regrid[index]
        del self._callbacks_slice[index]

    def state_add(self, state, regrid=True):
        self.states.append(state)
        self._callbacks_regrid.append(state.signal_regrid.connect(self.regrid))
        self._callbacks_slice.append(state.signal_slice.connect(self.reslice))
        assert state.ds == self.ds
        if regrid:
            self.regrid()

    def reslice(self, source_state=None):
        i1 = 0
        i2 = 0
        for state in self.states:
            subgrid = self.grid
            subgrid_sliced = self.grid
            axis = 0
            has_slice = False
            for other_state in self.states:
                if other_state == state: # simply skip these axes
                    for expression, shape, limit, slice_index in other_state.bin_parameters():
                        axis += 1
                else:
                    for expression, shape, limit, slice_index in other_state.bin_parameters():
                        if slice_index is not None:
                            subgrid_sliced = subgrid_sliced.__getitem__(tuple([slice(None)] * axis + [slice_index])).copy()
                            subgrid = np.sum(subgrid, axis=axis)
                            has_slice = True
                        else:
                            subgrid_sliced = np.sum(subgrid_sliced, axis=axis)
                            subgrid = np.sum(subgrid, axis=axis)
            state.grid = subgrid
            if has_slice:
                state.grid_sliced = subgrid_sliced
            else:
                state.grid_sliced = None
        

    def regrid(self, source_state=None):
        if not self.states:
            return
        binby = []
        shapes = []
        limits = []
        for state in self.states:
            for expression, shape, limit, slice_index in state.bin_parameters():
                binby.append(expression)
                limits.append(limit)
                shapes.append(shape)
#         print(binby, shapes, limits)
        self.grid = self.ds.count(binby=binby, shape=shapes, limits=limits)
        self.reslice()

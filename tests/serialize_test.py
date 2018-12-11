from common import *
import vaex.serialize

@vaex.serialize.register
class TestFunc:
    def state_get(self):
        return "hi"

def test_serialize_function():
    ds = vaex.from_scalars(x=1)
    test_func = TestFunc()
    ds.add_function('func', test_func)
    state = ds.state_get()
    assert 'functions' in state
    assert 'func' in state['functions']
    assert 'TestFunc' in state['functions']['func']['cls']
    assert 'hi' == state['functions']['func']['state']

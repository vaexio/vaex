import functools
import inspect
import vaex

from inspect import Signature, Parameter, signature

class Op:
    pass

register = {}

def generate_source_op(name, callable):
    class source_op(Op):
        def __init__(self, args, kwargs):
            self.args = args
            self.kwargs = kwargs
            self.name = name

        def execute(self):
            df = callable(*self.args, **self.kwargs)
            df.operation = self
            return df

        def to_json(self):
            sig = inspect.signature(callable)
            args = sig.bind(*self.args, **self.kwargs).arguments
            return dict(type='source', name=name, parameters=args)

    return source_op


def source(name):
    def f(callable):
        register[name] = generate_source_op(name, callable)
        def wrapper(*args, execute=True, **kwargs):
            op = register[name](args, kwargs)
            return op.execute() if execute else op
        # return wrapper
        return functools.wraps(callable)(wrapper)
    return f



def generate_transformation_op(name, callable):
    class transformation_op(Op):
        def __init__(self, child, args, kwargs):
            self.child = child
            self.args = args
            self.kwargs = kwargs
            self.name = name

        def execute(self, df):
            callable(df, *self.args, **self.kwargs)
            df.operation = self
            return df
        
        def to_json(self):
            # return dict(type='transformation', name=name, args=self.args, kwargs=self.kwargs, child=self.child.to_json())
            sig = inspect.signature(callable)
            args = sig.bind(None, *self.args, **self.kwargs).arguments
            args.pop('self')
            return dict(type='transformation', name=name, parameters=args, child=self.child.to_json())

        def __repr__(self):
            import sys
            import json
            data = self.to_json()
            json.dump(data, sys.stdout, indent=2)
            return ''


    return transformation_op


def transformation(name):
    def f(callable):
        register[name] = generate_transformation_op(name, callable)
        def wrapper(df, *args, execute=True, **kwargs):
            op = register[name](df.operation, args, kwargs)
            return op.execute(df) if execute else op
        return wrapper
        # return functools.wraps(callable, wrapper)
    return f



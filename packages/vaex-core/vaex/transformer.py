from abc import abstractmethod
import cloudpickle as pickle
import base64

import vaex.encoding


register = vaex.encoding.make_class_registery('transformer')


class Transformer:
    def __init__(self, previous):
        self.previous = previous

    def apply(self, df):
        return df

    def apply_deep(self, df):
        df = df.copy()
        df = self.previous.apply_deep(df) if self.previous else df
        return self.apply(df)


@register
class Apply(Transformer):
    snake_name = 'apply'
    def __init__(self, func, args, kwargs, previous):
        super().__init__(previous)
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        codeobj = base64.b64encode(pickle.dumps(self.func)).decode('ascii')
        code = ''
        if self.previous:
            code += f'{self.previous!r}'
        args = ", ".join(map(repr, self.args))
        kwargs = ", ".join(['{key}={value!r}' for key, value in self.kwargs.items()])
#         code += f'''import cloudpickle as pickle
# func = pickle.loads(base64.b64decode({codeobj!r}))
# '''
        code += 'func = lambda df, *args, **kwargs: df  # TODO: dummy function\n'
        if kwargs:
            code += f'df = df.transform(func, {args}, {self.kwargs})\n'
        else:
            code += f'df = df.transform(func, {args})\n'
        return code


    def apply(self, df):
        df = self.func(df, *self.args, **self.kwargs)
        if not isinstance(df, vaex.dataframe.DataFrame):
            raise ValueError('apply function should return a dataframe')
        return df._future()  # TODO: we can leave this out in v5

    def encode(self, encoding):
        return {
            'func': encoding.encode('pickle', self.func),
            'args': encoding.encode('pickle', self.args),
            'kwargs': encoding.encode('pickle', self.kwargs),
            'previous': encoding.encode('transformer', self.previous),
        }

    @classmethod
    def decode(cls, encoding, spec, trusted=False):
        if not trusted:
            raise ValueError("Will not unpickle func or arguments when source is not trusted")
        kwargs = {
            'func': encoding.decode('pickle', spec['func'], trusted=trusted),
            'args': encoding.decode('pickle', spec['args'], trusted=trusted),
            'kwargs': encoding.decode('pickle', spec['kwargs'], trusted=trusted),
            'previous': encoding.decode('transformer', spec['previous'], trusted=trusted),
        }
        return cls(**kwargs)

@register
class Method(Transformer):
    snake_name = 'method'
    def __init__(self, method_name, name, args, kwargs, previous):
        super().__init__(previous)
        self.method_name = method_name
        self.name = name
        def cast(name, value):
            if isinstance(value, vaex.expression.Expression):
                value = str(value)
            return value
        self.args = [cast(None, value) for value in args]
        self.kwargs = {name: cast(name, value) for name, value in kwargs.items()}

    def apply(self, df):
        method = getattr(df, self.method_name)  # TODO: check if allowed
        df = method(*self.args, **self.kwargs) or df
        return df

    def encode(self, encoding):
        return {
            'method_name': encoding.encode('json', self.method_name),
            'name': encoding.encode('json', self.name),
            'args': encoding.encode('json', self.args),
            'kwargs': encoding.encode('json', self.kwargs),
            'previous': encoding.encode('transformer', self.previous),
        }

    @classmethod
    def decode(cls, encoding, spec, trusted=False):
        kwargs = {
            'method_name': encoding.decode('json', spec['method_name']),
            'name': encoding.decode('json', spec['name']),
            'args': encoding.decode('json', spec['args']),
            'kwargs': encoding.decode('json', spec['kwargs']),
            'previous': encoding.decode('transformer', spec['previous'], trusted=trusted),
        }
        return cls(**kwargs)

    def __repr__(self):
        code = ''
        if self.previous:
            code += f'{self.previous!r}'
        args = ", ".join(map(repr, self.args))
        kwargs = ", ".join(['{key}={value!r}' for key, value in self.kwargs.items()])
        if kwargs:
            code += f'df = df.{self.method_name}({args}, {self.kwargs})\n'
        else:
            code += f'df = df.{self.method_name}({args})\n'
        return code

@register
class ML(Transformer):
    snake_name = 'ml'
    def __init__(self, ml_transformer, previous):
        super().__init__(previous)
        self.ml_transformer = ml_transformer

    def apply(self, df):
        df = self.ml_transformer.transform(df)
        return df

    def encode(self, encoding):
        return {
            'state': vaex.serialize.to_dict(self.ml_transformer),
            'previous': encoding.encode('transformer', self.previous),
        }

    @classmethod
    def decode(cls, encoding, spec, trusted=False):
        kwargs = {
            'ml_transformer': vaex.serialize.from_dict(spec['state']),
            'previous': encoding.decode('transformer', spec['previous'], trusted=trusted),
        }
        return cls(**kwargs)

    def __repr__(self):
        code = ''
        if self.previous:
            code += f'{self.previous!r}'
        # args = ", ".join(map(repr, self.args))
        # kwargs = ", ".join(['{key}={value!r}' for key, value in self.kwargs.items()])
        # if kwargs:
        #     code += f'df = df.{self.method_name}({args}, {self.kwargs})\n'
        # else:
        #     code += f'df = df.{self.method_name}({args})\n'
        code += f"ml_transformer = None # TODO, fix this {type(self.ml_transformer)}\n"
        code += 'df = ml_transformer.transform(df)\n'
        return code
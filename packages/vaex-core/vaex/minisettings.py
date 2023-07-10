import os
from pathlib import Path
from typing import Any, Optional

# similar API to pydantic/pydantic-settings but we prefer not to have a dependency on pydantic
# since we cannot be compatible with pydantic1 and 2
# NOTE: not a public api


def _get_type(annotation):
    check_optional_types = [str, int, float, bool, dict, list]
    for check_type in check_optional_types:
        if annotation == Optional[check_type]:
            return check_type
    if hasattr(annotation, "__origin__"):
        if annotation.__origin__ == dict:
            return dict
    return annotation


class _Field:
    def __init__(self, default=None, env=None, title=None, default_factory=None, gt=None, alias=None) -> None:
        self.default = default
        self.env = env
        self.fullenv = None
        self.title = title
        self.annotation = None
        self.default_factory = default_factory
        self.gt = gt
        self.alias = alias
        self.field_info = self
        self.extra = {"env_names": [env] if env else []}

    def __set_name__(self, owner, name):
        prefix = "SOLARA_"
        config = getattr(owner, "Config")
        if config:
            prefix = getattr(config, "env_prefix", prefix).upper()
            if hasattr(config, "fields"):
                fields = config.fields
                if name in fields:
                    self.alias = fields[name]
        self.name = name
        self.alias = self.alias or self.name
        self.title = self.title or self.name
        if self.env is None:
            self.env = f"{prefix}{self.name.upper()}"
        else:
            self.env = self.env
        self.annotation = owner.__annotations__.get(self.name)
        assert self.annotation is not None, f"Field {self.name} must have a type annotation"
        self.type_ = _get_type(self.annotation)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance._values[self.name]


def convert(annotation, value: str) -> Any:
    check_optional_types = [str, int, float, bool, Path]
    for check_type in check_optional_types:
        if annotation == Optional[check_type]:
            annotation = check_type
            return convert(annotation, value)
    if annotation == str:
        return value
    elif annotation == int:
        return int(value)
    elif annotation == float:
        return float(value)
    elif annotation == bool:
        if value in ("True", "true", "1"):
            return True
        elif value in ("False", "false", "0"):
            return False
        else:
            raise ValueError(f"Invalid boolean value {value}")
    else:
        # raise TypeError(f"Unsupported type {annotation}")
        return annotation(value)


def Field(*args, **kwargs) -> Any:
    return _Field(*args, **kwargs)


class BaseSettings:
    __fields__: dict

    def __init__(self, **kwargs) -> None:
        cls = type(self)
        self._values = {**kwargs}
        keys = set([k.upper() for k in os.environ.keys()])
        for key, field in cls.__dict__.items():
            if key in kwargs:
                continue
            if isinstance(field, _Field):
                value = field.default
                if field.default_factory:
                    value = field.default_factory()

                if field.env:
                    env_key = field.env.upper()
                    if env_key in keys:
                        # do a case-insensitive lookup
                        for env_var_cased in os.environ.keys():
                            if env_key.upper() == env_var_cased.upper():
                                value = convert(field.annotation, os.environ[env_var_cased])
                self._values[key] = value

    def __init_subclass__(cls) -> None:
        cls.__fields__ = {}
        for key, field in cls.__dict__.items():
            if key.startswith("_"):
                continue
            if key == "Config":
                continue
            if not isinstance(field, _Field):
                field = Field(field)
                setattr(cls, key, field)
                field.__set_name__(cls, key)
            cls.__fields__[key] = field

    def dict(self, by_alias=True):
        values = self._values.copy()
        for key, value in values.items():
            if isinstance(value, BaseSettings):
                values[key] = value.dict(by_alias=by_alias)
        return values

import json
import os
import sys

import traitlets

from . import sklearn
from . import generate
from . import catboost
from . import lightgbm
from . import xgboost
from . import cluster
from .incubator import river


def lmap(f, values):
    return list(map(f, values))


def lmapstar(f, values):
    return [f(*k) for k in values]


def to_trait(name, trait):
    return dict(
        name=name,
        has_default=trait.default_value is traitlets.Undefined,
        default=None
        if trait.default_value is traitlets.Undefined
        else trait.default_value,
        type=str(type(trait).__name__),
        help=trait.help,
    )


def to_cls(cls):
    return dict(
        classname=cls.__name__,
        snake_name=cls.__dict__.get(
            "snake_name", generate.camel_to_underscore(cls.__name__)
        ),
        version=cls.__dict__.get("_version", "1.0.0"),
        module=cls.__module__,
        traits=lmapstar(to_trait, cls.class_traits().items()),
        doc=cls.__doc__
    )


def main(args=sys.argv):
    spec = lmap(to_cls, generate.registry)
    json_data = json.dumps(spec, indent=4, sort_keys=True)
    path = os.path.join(os.path.dirname(__file__), "spec.json")
    if len(sys.argv) > 1:
        path = sys.argv[1]
    with open(path, "w") as f:
        f.write(json_data)


if __name__ == "__main__":
    main()

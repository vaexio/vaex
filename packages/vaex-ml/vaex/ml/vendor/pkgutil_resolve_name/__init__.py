"""
Resolve a name to an object.

It is expected that `name` will be a string in one of the following
formats, where W is shorthand for a valid Python identifier and dot stands
for a literal period in these pseudo-regexes:

W(.W)*
W(.W)*:(W(.W)*)?

The first form is intended for backward compatibility only. It assumes that
some part of the dotted name is a package, and the rest is an object
somewhere within that package, possibly nested inside other objects.
Because the place where the package stops and the object hierarchy starts
can't be inferred by inspection, repeated attempts to import must be done
with this form.

In the second form, the caller makes the division point clear through the
provision of a single colon: the dotted name to the left of the colon is a
package to be imported, and the dotted name to the right is the object
hierarchy within that package. Only one import is needed in this form. If
it ends with the colon, then a module object is returned.

The function will return an object (which might be a module), or raise one
of the following exceptions:

ValueError - if `name` isn't in a recognised format
ImportError - if an import failed when it shouldn't have
AttributeError - if a failure occurred when traversing the object hierarchy
                 within the imported package to get to the desired object)
"""

import importlib
import re

__version__ = "1.0.0"

_DOTTED_WORDS = r'[a-z_]\w*(\.[a-z_]\w*)*'
_NAME_PATTERN = re.compile('^({_DOTTED_WORDS})(:({_DOTTED_WORDS})?)?$'.format(_DOTTED_WORDS=_DOTTED_WORDS), re.I)
del _DOTTED_WORDS


def resolve_name(name):
    m = _NAME_PATTERN.match(name)
    if not m:
        raise ValueError('invalid format: {name!r}'.format(name=name))
    groups = m.groups()
    if groups[2]:
        # there is a colon - a one-step import is all that's needed
        mod = importlib.import_module(groups[0])
        parts = groups[3].split('.') if groups[3] else []
    else:
        # no colon - have to iterate to find the package boundary
        parts = name.split('.')
        modname = parts.pop(0)
        # first part *must* be a module/package.
        mod = importlib.import_module(modname)
        while parts:
            p = parts[0]
            s = '{modname}.{p}'.format(modname=modname, p=p)
            try:
                mod = importlib.import_module(s)
                parts.pop(0)
                modname = s
            except ImportError:
                break
    # if we reach this point, mod is the module, already imported, and
    # parts is the list of parts in the object hierarchy to be traversed, or
    # an empty list if just the module is wanted.
    result = mod
    for p in parts:
        result = getattr(result, p)
    return result

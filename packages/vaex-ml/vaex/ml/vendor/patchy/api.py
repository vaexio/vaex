import __future__

import ast
import inspect
import os
import shutil
import subprocess
import sys
from functools import wraps
from tempfile import mkdtemp
from textwrap import dedent
from weakref import WeakKeyDictionary

from .cache import PatchingCache

if sys.version_info >= (3, 9):  # pragma: no cover
    from pkgutil import resolve_name as pkgutil_resolve_name
else:  # pragma: no cover
    from pkgutil_resolve_name import resolve_name as pkgutil_resolve_name

__all__ = ("patch", "mc_patchface", "unpatch", "replace", "temp_patch")


# Public API


def patch(func, patch_text):
    return _do_patch(func, patch_text, forwards=True)


mc_patchface = patch


def unpatch(func, patch_text):
    return _do_patch(func, patch_text, forwards=False)


def replace(func, expected_source, new_source):
    if expected_source is not None:
        expected_source = dedent(expected_source)
        current_source = _get_source(func)
        _assert_ast_equal(current_source, expected_source, func.__name__)

    new_source = dedent(new_source)
    _set_source(func, new_source)


class temp_patch:
    def __init__(self, func, patch_text):
        self.func = func
        self.patch_text = patch_text

    def __enter__(self):
        patch(self.func, self.patch_text)

    def __exit__(self, _, __, ___):
        unpatch(self.func, self.patch_text)

    def __call__(self, decorable):
        @wraps(decorable)
        def wrapper(*args, **kwargs):
            with self:
                decorable(*args, **kwargs)

        return wrapper


# Gritty internals


def _do_patch(func, patch_text, forwards):
    if isinstance(func, str):
        func = pkgutil_resolve_name(func)
    source = _get_source(func)
    patch_text = dedent(patch_text)

    new_source = _apply_patch(source, patch_text, forwards, func.__name__)

    _set_source(func, new_source)


_patching_cache = PatchingCache(maxsize=100)


def _apply_patch(source, patch_text, forwards, name):
    # Cached ?
    try:
        return _patching_cache.retrieve(source, patch_text, forwards)
    except KeyError:
        pass

    # Write out files
    tempdir = mkdtemp(prefix="patchy")
    try:
        source_path = os.path.join(tempdir, name + ".py")
        with open(source_path, "w") as source_file:
            source_file.write(source)

        patch_path = os.path.join(tempdir, name + ".patch")
        with open(patch_path, "w") as patch_file:
            patch_file.write(patch_text)
            if not patch_text.endswith("\n"):
                patch_file.write("\n")

        # Call `patch` command
        command = ["patch"]
        if not forwards:
            command.append("--reverse")
        command.extend([source_path, patch_path])
        proc = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            msg = "Could not {action} the patch {prep} '{name}'.".format(
                action=("apply" if forwards else "unapply"),
                prep=("to" if forwards else "from"),
                name=name,
            )
            msg += " The message from `patch` was:\n{}\n{}".format(
                stdout.decode("utf-8"), stderr.decode("utf-8")
            )
            msg += "\nThe code to patch was:\n{}\nThe patch was:\n{}".format(
                source, patch_text
            )
            raise ValueError(msg)

        with open(source_path) as source_file:
            new_source = source_file.read()
    finally:
        shutil.rmtree(tempdir)

    _patching_cache.store(source, patch_text, forwards, new_source)

    return new_source


def _get_flags_mask():
    result = 0
    for name in __future__.all_feature_names:
        result |= getattr(__future__, name).compiler_flag
    return result


FEATURE_MASK = _get_flags_mask()


# Stores the source of functions that have had their source changed
_source_map = WeakKeyDictionary()


def _get_source(func):
    real_func = _get_real_func(func)
    try:
        return _source_map[real_func]
    except KeyError:
        source = inspect.getsource(func)
        source = dedent(source)
        return source


def _class_name(func):
    qualname = getattr(func, "__qualname__", None)
    if qualname is not None:  # pragma: no py2 cover
        split_name = qualname.split(".")
        try:
            class_name = split_name[-2]
        except IndexError:
            return None
        else:
            if class_name == "<locals>":
                return None
            return class_name
    else:  # pragma: no py3 cover
        im_class = getattr(func, "im_class", None)
        if im_class is not None:
            return im_class.__name__


def _set_source(func, func_source):
    # Fetch the actual function we are changing
    real_func = _get_real_func(func)
    # Figure out any future headers that may be required
    feature_flags = real_func.__code__.co_flags & FEATURE_MASK

    class_name = _class_name(func)

    def _compile(code, flags=0):
        return compile(
            code, "<patchy>", "exec", flags=feature_flags | flags, dont_inherit=True
        )

    def _parse(code):
        return _compile(code, flags=ast.PyCF_ONLY_AST)

    def _process_freevars():
        """
        Wrap the new function in a __patchy_freevars__ method that provides all
        freevars of the original function.

        Because the new function must use exectaly the same freevars as the
        original, also append to the new function with a body of code to force
        use of those freevars (in the case the the patch drops use of any
        freevars):

        def __patchy_freevars__():
            eg_free_var_spam = object()  <- added in wrapper
            eg_free_var_ham = object()   <- added in wrapper

            def patched_func():
                return some_global(eg_free_var_ham)
                eg_free_var_spam         <- appended to new func body
                eg_free_var_ham          <- appended to new func body

            return patched_func
        """
        _def = "def __patchy_freevars__():"
        fvs = func.__code__.co_freevars
        fv_body = ["    {} = object()".format(fv) for fv in fvs]
        fv_force_use_body = ["    {}".format(fv) for fv in fvs]
        if fv_force_use_body:
            fv_force_use_ast = _parse("\n".join([_def] + fv_force_use_body))
            fv_force_use = fv_force_use_ast.body[0].body
        else:
            fv_force_use = []
        _ast = _parse(func_source).body[0]
        _ast.body = _ast.body + fv_force_use
        return _def, _ast, fv_body

    def _process_method():
        """
        Wrap the new method in a class to ensure the same mangling as would
        have been performed on the original method:

        def __patchy_freevars__():

            class SomeClass(object):
                def patched_func(self):
                    return some_globals(self.__some_mangled_prop)

            return SomeClass.patched_func
        """

        _def, _ast, fv_body = _process_freevars()
        _global = (
            ""
            if class_name in func.__code__.co_freevars
            else "    global {name}\n".format(name=class_name)
        )
        class_src = "{_global}    class {name}(object):\n        pass".format(
            _global=_global, name=class_name
        )
        ret = "    return {class_name}.{name}".format(
            class_name=class_name, name=func.__name__
        )
        to_parse = "\n".join([_def] + fv_body + [class_src, ret])
        new_source = _parse(to_parse)
        new_source.body[0].body[-2].body[0] = _ast
        return new_source

    def _process_function():
        _def, _ast, fv_body = _process_freevars()
        name = func.__name__
        ret = "    return {name}".format(name=name)
        _global = (
            []
            if name in func.__code__.co_freevars
            else ["    global {name}".format(name=name)]
        )
        to_parse = "\n".join([_def] + _global + fv_body + ["    pass", ret])
        new_source = _parse(to_parse)
        new_source.body[0].body[-2] = _ast
        return new_source

    if class_name:
        new_source = _process_method()
    else:
        new_source = _process_function()

    # Compile and retrieve the new Code object
    localz = {}
    new_code = _compile(new_source)

    exec(new_code, dict(func.__globals__), localz)
    new_func = localz["__patchy_freevars__"]()

    # Figure out how to get the Code object
    if isinstance(new_func, (classmethod, staticmethod)):  # pragma: no py3 cover
        new_code = new_func.__func__.__code__
    else:
        new_code = new_func.__code__

    # Put the new Code object in place
    real_func.__code__ = new_code
    # Store the modified source. This used to be attached to the function but
    # that is a bit naughty
    _source_map[real_func] = func_source


def _get_real_func(func):
    """
    Duplicates some of the logic implicit in inspect.getsource(). Basically
    some function-esque things, such as classmethods, aren't functions but we
    can peel back the layers to the underlying function very easily.
    """
    if inspect.ismethod(func):
        return func.__func__
    else:
        return func


def _assert_ast_equal(current_source, expected_source, name):
    current_ast = ast.parse(current_source)
    expected_ast = ast.parse(expected_source)
    if not ast.dump(current_ast) == ast.dump(expected_ast):
        msg = (
            "The code of '{name}' has changed from expected.\n"
            "The current code is:\n{current_source}\n"
            "The expected code is:\n{expected_source}"
        ).format(
            name=name, current_source=current_source, expected_source=expected_source
        )
        raise ValueError(msg)

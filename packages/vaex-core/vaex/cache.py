import functools
import numpy
import dask.base
import logging

log = logging.getLogger('vaex.cache')


import vaex.utils

def tokenize(*args, **kwargs):
    return dask.base.tokenize(*args, **kwargs)


def output_file(callable=None, path_input=None, fs_options_input={}, path_output=None, fs_options_output={}):
    """Decorator to do cached conversion from path_input → path_output.

    Caching is active if with the input file, the *args, and **kwargs are unchanged, otherwise callable
    will be called again.


    """
    def wrapper1(callable=callable):
        def wrapper2(callable=callable):
            def call(callable=callable, *args, **kwargs):
                base, ext, fs_options_output_ = vaex.file.split_ext(path_output, fs_options_output)
                path_output_meta = base + '.yaml'
                # this token changes if input changes, or args, or kwargs
                token = tokenize(vaex.file.tokenize(path_input, fs_options_input), args, kwargs)

                def write_token():
                    log.info('saving token %r for %s → %s conversion to %s', token, path_input, path_output, path_output_meta)
                    with vaex.file.open(path_output_meta, 'w', fs_options=fs_options_output_) as f:
                        f.write(f"# this file exists so that we know when not to do the\n# {path_input} → {path_output} conversion\n")
                        vaex.utils.yaml_dump(f, {'token': token})

                if not vaex.file.exists(path_output, fs_options=fs_options_output_):
                    log.info('file %s does not exist yet, running conversion %s → %s', path_output_meta, path_input, path_output)
                    value = callable(*args, **kwargs)
                    write_token()
                    return value

                if not vaex.file.exists(path_output_meta, fs_options=fs_options_output_):
                    log.info('file including token not found (%) or does not exist yet, running conversion %s → %s', path_output_meta, path_input, path_output)
                    value = callable(*args, **kwargs)
                    write_token()
                    return value

                # load token
                with vaex.file.open(path_output_meta, fs_options=fs_options_output_) as f:
                    output_meta = vaex.utils.yaml_load(f)

                if output_meta['token'] != token:
                    log.info('token for %s is out of date, rerunning conversion to %s', path_input, path_output)
                    value = callable(*args, **kwargs)
                    write_token()
                    return value
                else:
                    log.info('token for %s did not change, reusing converted file %s', path_input, path_output)
            return call
        return wrapper2
    return wrapper1()

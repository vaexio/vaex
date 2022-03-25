import os
import logging
import vaex.utils
import collections

import json
import multiprocessing
import sys

from pydantic import BaseModel, BaseSettings, Field
from typing import List, Union, Optional, Dict
from enum import Enum
from .config import ConfigDefault

logger = logging.getLogger("vaex.settings")
_default_home = vaex.utils.get_vaex_home()
try:
    import dotenv
    has_dotenv = True
except:
    has_dotenv = False
if has_dotenv:
    from pydantic.env_settings import read_env_file
    envs = read_env_file(ConfigDefault.env_file)
    _default_home = envs.get('vaex_home', _default_home)

# we may want to use this
# class ByteAmount(str):
#     @classmethod
#     def validate(cls, value):
#         try:
#             import ast
#             value = ast.literal_eval(value)
#         except:
#             pass
#         if isinstance(value, str):
#             import dask
#             value = dask.utils.parse_bytes(value)
#         if not isinstance(value, int):
#             raise TypeError(f"Expected a number of bytes, got {value}")
#         return value

#     def __repr__(self):
#         return f'PostCode({super().__repr__()})'


class MemoryTrackerEnum(str, Enum):
    default = 'default'
    limit = 'limit'


class MemoryTracker(BaseSettings):
    """Memory tracking/protection when using vaex in a service"""
    type: MemoryTrackerEnum = Field('default', title="Which memory tracker to use when executing tasks", env="VAEX_MEMORY_TRACKER")
    max: Optional[str] = Field(None, title="How much memory the executor can use maximally (only used for type='limit')")
    class Config:
        use_enum_values = True
        env_prefix = 'vaex_memory_tracker_'



class TaskTracker(BaseSettings):
    """task tracking/protection when using vaex in a service"""
    type: str = Field('', title="Comma seperated string of trackers to run while executing tasks", env="VAEX_TASK_TRACKER")
    class Config:
        use_enum_values = True
        env_prefix = 'vaex_task_tracker_'


class Display(BaseSettings):
    """How a dataframe displays"""
    max_columns: int = Field(200, title="How many column to display when printing out a dataframe")
    max_rows: int = Field(10, title="How many rows to print out before showing the first and last rows")
    class Config(ConfigDefault):
        env_prefix = 'vaex_display_'


class Chunk(BaseSettings):
    """Configure how a dataset is broken down in smaller chunks. The executor dynamically adjusts the chunk size based on `size_min` and `size_max` and the number of threads when `size` is not set."""
    size: Optional[int] = Field(title="When set, fixes the number of chunks, e.g. do not dynamically adjust between min and max")
    size_min: int = Field(1024, title="Minimum chunk size")
    size_max: int = Field(1024**2, title="Maximum chunk size")
    class Config(ConfigDefault):
        env_prefix = 'vaex_chunk_'


class Cache(BaseSettings):
    """Setting for caching of computation or task results, see the [API](api.html#module-vaex.cache) for more details."""
    type: Optional[str] = Field(None, env='VAEX_CACHE', title="Type of cache, e.g. 'memory_infinite', 'memory', 'disk', 'redis', or a multilevel cache, e.g. 'memory,disk'")
    disk_size_limit: str = Field('10GB', title='Maximum size for cache on disk, e.g. 10GB, 500MB')
    memory_size_limit: str = Field('1GB', title='Maximum size for cache in memory, e.g. 1GB, 500MB')
    path: Optional[str] = Field(os.path.join(_default_home, "cache"), env="VAEX_CACHE_PATH", title="Storage location for cache results. Defaults to `${VAEX_HOME}/cache`")

    class Config(ConfigDefault):
        env_prefix = 'vaex_cache_'


class AsyncEnum(str, Enum):
    nest = 'nest'
    awaitio = 'awaitio'

try:
    import vaex.server
    has_server = True
except ImportError:
    has_server = False

if has_server:
    import vaex.server.settings


class FileSystem(BaseSettings):
    """Filesystem configuration"""
    path: str = Field(os.path.join(_default_home, "file-cache"), env="VAEX_FS_PATH", title="Storage location for caching files from remote file systems. Defaults to `${VAEX_HOME}/file-cache/`")

    class Config(ConfigDefault):
        env_prefix = 'vaex_fs_'

class Data(BaseSettings):
    """Data configuration"""
    path: str = Field(os.path.join(_default_home, "data"), env="VAEX_DATA_PATH", title="Storage location for data files, like vaex.example(). Defaults to `${VAEX_HOME}/data/`")

    class Config(ConfigDefault):
        env_prefix = 'vaex_data_'

class Progress(BaseSettings):
    """Data configuration"""
    type: str = Field('simple', title="Default progressbar to show: 'simple', 'rich' or 'widget'")
    force: str = Field(None, title="Force showing a progress bar of this type, even when no progress bar was requested from user code", env="VAEX_PROGRESS")

    class Config(ConfigDefault):
        env_prefix = 'vaex_progress_'


class Logging(BaseSettings):
    """Configure logging for Vaex. By default Vaex sets up logging, which is useful when running a script. When Vaex is used in applications or services that already configure logging, set the environomental variables VAEX_LOGGING_SETUP to false.

See the [API docs](api.html#module-vaex.logging) for more details.

Note that settings `vaex.settings.main.logging.info` etc at runtime, has no direct effect, since logging is already configured. When needed, call `vaex.logging.reset()` and `vaex.logging.setup()` to reconfigure logging.
    """
    setup : bool = Field(True, title='Setup logging for Vaex at import time.')
    rich : bool = Field(True, title='Use rich logger (colored fancy output).')
    debug : str = Field('', title="Comma seperated list of loggers to set to the debug level (e.g. 'vaex.settings,vaex.cache'), or a '1' to set the root logger ('vaex')")
    info : str = Field('', title="Comma seperated list of loggers to set to the info level (e.g. 'vaex.settings,vaex.cache'), or a '1' to set the root logger ('vaex')")
    warning : str = Field('vaex', title="Comma seperated list of loggers to set to the warning level (e.g. 'vaex.settings,vaex.cache'), or a '1' to set the root logger ('vaex')")
    error : str = Field('', title="Comma seperated list of loggers to set to the error level (e.g. 'vaex.settings,vaex.cache'), or a '1' to set the root logger ('vaex')")
    class Config(ConfigDefault):
        env_prefix = 'vaex_logging_'


class Settings(BaseSettings):
    """General settings for vaex"""
    aliases: Optional[dict] = Field(title='Aliases to be used for vaex.open', default_factory=dict)
    async_: AsyncEnum = Field('nest', env='VAEX_ASYNC', title="How to run async code in the local executor", min_length=2)
    home: str = Field(_default_home, env="VAEX_HOME", title="Home directory for vaex, which defaults to `$HOME/.vaex`, "\
        " If both `$VAEX_HOME` and `$HOME` are not defined, the current working directory is used. (Note that this setting cannot be configured from the vaex home directory itself).")
    mmap: bool = Field(True, title="Experimental to turn off, will avoid using memory mapping if set to False")
    process_count: Optional[int] = Field(title="Number of processes to use for multiprocessing (e.g. apply), defaults to thread_count setting", gt=0)
    thread_count: Optional[int] = Field(env='VAEX_NUM_THREADS', title="Number of threads to use for computations, defaults to multiprocessing.cpu_count()", gt=0)
    thread_count_io: Optional[int] = Field(env='VAEX_NUM_THREADS_IO', title="Number of threads to use for IO, defaults to thread_count_io + 1", gt=0)
    path_lock: str = Field(os.path.join(_default_home, "lock"), env="VAEX_LOCK", title="Directory to store lock files for vaex, which defaults to `${VAEX_HOME}/lock/`, "\
        " Due to possible race conditions lock files cannot be removed while processes using Vaex are running (on Unix systems).")


    # avoid name collisions of VAEX_CACHE with configurting the whole object via json in env var
    cache = Field(Cache(), env='_VAEX_CACHE')
    chunk: Chunk = Field(Chunk(), env='_VAEX_CHUNK')
    data = Field(Data(), env='_VAEX_DATA')
    display: Display = Field(Display(), env='_VAEX_DISPLAY')
    fs: FileSystem = Field(FileSystem(), env='_VAEX_FS')
    memory_tracker = Field(MemoryTracker(), env='_VAEX_MEMORY_TRACKER')
    task_tracker = Field(TaskTracker(), env='_VAEX_TASK_TRACKER')
    logging = Field(Logging(), env="_VAEX_LOGGING")
    progress = Field(Progress(), env="_VAEX_PROGRESS")

    if has_server:
        server: vaex.server.settings.Settings = vaex.server.settings.Settings()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # dynamic defaults
        if self.thread_count is None:
            self.thread_count = multiprocessing.cpu_count()
            self.__fields__['thread_count'].default = self.thread_count
        if self.thread_count_io is None:
            self.thread_count_io = self.thread_count + 1
            self.__fields__['thread_count_io'].default = self.thread_count
        if self.process_count is None:
            self.process_count = self.thread_count
            self.__fields__['process_count'].default = self.thread_count

    class Config(ConfigDefault):
        use_enum_values = True
        allow_population_by_field_name = True
        allow_population_by_alias = True
        fields = {
            'async_': 'async'
        }
        case_sensitive = False
        env_prefix = 'vaex_'

_default_values = {}
filename = os.path.join(vaex.utils.get_vaex_home(), "main.yml")
if os.path.exists(filename):
    with open(filename) as f:
        _default_values = vaex.utils.yaml_load(f)
    if _default_values is None:
        _default_values = {}


main = Settings(**_default_values)
if has_server:
    server = main.server
display = main.display
fs = main.fs
cache = main.cache
aliases = main.aliases
data = main.data


def save(exclude_defaults=True, verbose=False):
    filename = os.path.join(vaex.utils.get_private_dir(), "main.yml")
    if verbose:
        values = main.dict(by_alias=True)
        print("All values:\n")
        vaex.utils.yaml_dump(sys.stdout, values)

    with open(filename, "w") as f:
        values = main.dict(by_alias=True, exclude_defaults=exclude_defaults)
        vaex.utils.yaml_dump(f, values)
        if verbose:
            print("Saved values:\n")
            vaex.utils.yaml_dump(sys.stdout, values)


def edit_jupyter():
    import vaex.jupyter.widgets
    editor = vaex.jupyter.widgets.SettingsEditor(schema=main.schema(), values=main.dict())
    return editor


def _to_md(cls, f=sys.stdout):
    printf = lambda *x: print(*x, file=f)
    title = cls.__name__
    printf(f"## {title}")
    printf(cls.__doc__)
    printf()

    # own fields
    for name, field in cls.__fields__.items():
        pyname  = name
        name = field.alias
        if issubclass(field.type_, BaseSettings):
            continue
        printf(f"### {name}")
        title = field.field_info.title
        if title is None:
            raise ValueError(f'Title missing for {name}')
        env_name = (cls.Config.env_prefix + name).upper()
        if field.field_info.extra["env_names"]:
            if len(field.field_info.extra["env_names"]) > 1:
                raise NotImplementedError('should we support this?')
            env_name = list(field.field_info.extra["env_names"])[0].upper()
        default = field.default

        printf(title)
        printf()

        printf(f'Environmental variable: `{env_name}`')
        if default is not None:
            printf()
            printf(f'Example use:\n ```\n$ {env_name}={default} python myscript.py\n```')
        printf()
        flat = {
            Settings: 'main',
            Chunk: 'main.chunk',
            Display: 'display',
            vaex.server.settings.Settings: 'server',
            Cache: 'cache',
            MemoryTracker: 'main.memory_tracker',
            TaskTracker: 'main.task_tracker',
            Cache: 'cache',
            FileSystem: 'fs',
            Data: 'data',
            Logging: 'main.logging',
            Progress: 'main.progress',
        }[cls]
        pyvar = f'vaex.settings.{flat}.{pyname}'
        printf(f'Python settings `{pyvar}`')
        if default not in [None, ""]:
            printf()
            printf(f'Example use: `{pyvar} = {default!r}`')

    # sub objects
    for name, field in cls.__fields__.items():
        pyname  = name
        name = field.alias
        if not issubclass(field.type_, BaseSettings):
            continue
        name = field.type_.__name__
        _to_md(field.type_, f=f)
        printf()


def _watch():
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

    class EventHandler(FileSystemEventHandler):
        def on_modified(self, event):
            super(EventHandler, self).on_modified(event)
            print(f"Change detected, running docgen...")
            if os.system("vaex settings docgen") == 0:
                print("done")
            else:
                print("error")
    observer = Observer()
    path = __file__
    print(f"Running first time")
    os.system("vaex settings docgen")
    print(f"Watching {path}")
    observer.schedule(EventHandler(), path, recursive=True)
    observer.start()
    observer.join()


def _main(args):
    if len(args) > 1:
        type = args[1]
        if type == "schema":
            print(main.schema_json(indent=2))
        elif type == "yaml":
            values = main.dict(by_alias=True)
            vaex.utils.yaml_dump(sys.stdout, values)
        elif type == "yaml-diff":
            values = main.dict(by_alias=True, exclude_defaults=True)
            vaex.utils.yaml_dump(sys.stdout, values)
        elif type == "json":
            values = main.dict(by_alias=True)
            json.dump(values, sys.stdout, indent=2)
        elif type == "save":
            save(exclude_defaults=True, verbose=True)
        elif type == "set":
            save(exclude_defaults=True, verbose=True)
        elif type == "save-defaults":
            save(exclude_defaults=False, verbose=True)
        elif type == "md":
            _to_md(Settings)
        elif type == "watch":
            # runs docgen automatically
            _watch()
        elif type == "docgen":
            with open('docs/source/conf.md') as f:
                current_md = f.read()
            import io
            f = io.StringIO()
            _to_md(Settings, f=f)
            code = f.getvalue()
            marker = '<!-- autogenerate markdown below -->'
            start = current_md.find(marker)
            start = current_md.find('\n', start)
            with open('docs/source/conf.md', 'w') as f:
                print(current_md[:start+1], file=f)
                print(code, file=f)
        else:
            raise ValueError('only support schema, values, save, save-defaults or md')
    else:
        print(json.dumps(main.dict(by_alias=True), indent=2))



if __name__ == "__main__":
    _main(sys.argv)

import os
import math
import threading

import vaex.settings

try:
    from sys import version_info
    if version_info[:2] >= (3, 10):
        from importlib.metadata import entry_points
    else:
        from importlib_metadata import entry_points
except ImportError:
    import pkg_resources
    entry_points = pkg_resources.iter_entry_points

# thread local variable, where 'global' tracker go
local = threading.local()
_memory_tracker_types = {}
lock = threading.Lock()



class MemoryTracker:
    track_live = False
    def __init__(self) -> None:
        self.used = 0

    def pre_alloc(self, bytes, reason):
        self.used += bytes

    def using(self, bytes):
        self.used += bytes


def create_tracker():
    memory_tracker_type = vaex.settings.main.memory_tracker.type
    if not _memory_tracker_types:
        with lock:
            if not _memory_tracker_types:
                for entry in entry_points(group="vaex.memory.tracker"):
                    _memory_tracker_types[entry.name] = entry.load()
    cls = _memory_tracker_types.get(memory_tracker_type)
    if cls is not None:
        return cls()
    raise ValueError(f"No memory tracker found with name {memory_tracker_type}")

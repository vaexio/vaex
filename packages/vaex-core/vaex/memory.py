import os
import math
import threading

import pkg_resources
import vaex.settings

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
                for entry in pkg_resources.iter_entry_points(group="vaex.memory.tracker"):
                    _memory_tracker_types[entry.name] = entry.load()
    cls = _memory_tracker_types.get(memory_tracker_type)
    if cls is not None:
        return cls()
    raise ValueError(f"No memory tracker found with name {memory_tracker_type}")

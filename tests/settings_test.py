import subprocess
import os
import contextlib


@contextlib.contextmanager
def env(name, value):
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is not None:
            os.environ[previous]


def test_settings():
    output = subprocess.check_output(['vaex', 'settings']).decode('utf8')
    assert "memory,disk" not in output
    with env('VAEX_CACHE', "memory,disk"):
        output = subprocess.check_output(['vaex', 'settings']).decode('utf8')
        assert "memory,disk" in output

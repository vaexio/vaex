import os
import pytest


# Creating a custom mark decorator for units that test belong the incubator.
skip_incubator = pytest.mark.skipif('RUN_INCUBATOR_TESTS' not in os.environ,
                                    reason="Add environment variable RUN_INCUBATOR_TESTS to run this test since \
                                    modules and libraries in the incubator may change abruplty without notice.")


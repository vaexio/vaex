import os
from pydantic import BaseSettings
import vaex.utils


# in seperate module to avoid circular imports

class ConfigDefault:
    env_file = ".env"
    @classmethod
    def customise_sources(cls, init_settings, env_settings, file_secret_settings):
        return (
            env_settings,
            file_secret_settings,
            init_settings,  # constructor argument last, since they come from global yaml file
        )

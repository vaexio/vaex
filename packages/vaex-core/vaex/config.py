import os
import vaex.utils
import pydantic

# in seperate module to avoid circular imports
# with pydantic 2.0, we require pydantic_settings
try:
    import pydantic_settings
except ModuleNotFoundError:
    # we should be on pydantic 1.x
    BaseSettings = pydantic.BaseSettings
else:
    major = pydantic_settings.__version__.split(".")[0]
    if major != "0":
        # but the old pydantic_settings is unrelated
        BaseSettings = pydantic_settings.BaseSettings
    else:
        # we should be on pydantic 2.x
        BaseSettings = pydantic.BaseSettings

class ConfigDefault:
    env_file = ".env"
    @classmethod
    def customise_sources(cls, init_settings, env_settings, file_secret_settings):
        return (
            env_settings,
            file_secret_settings,
            init_settings,  # constructor argument last, since they come from global yaml file
        )


from typing import Dict
from pydantic import Field
import pydantic
import vaex.config

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


class Settings(BaseSettings):
    '''Configuration options for the FastAPI server'''
    # vaex_config_file: str = "vaex-server.json"
    add_example: bool = Field(True, title="Add example dataset")
    # vaex_config: dict = None
    graphql: bool = Field(False, title="Add graphql endpoint")
    files: Dict[str, str] = Field(default_factory=dict, title="Mapping of name to path")
    class Config(vaex.config.ConfigDefault):
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_prefix = 'vaex_server_'
        # secrets_dir = '/run/secrets'

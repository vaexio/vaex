
from typing import Dict
from pydantic import BaseSettings, Field
import vaex.config

class Settings(BaseSettings):
    '''Configuration options for the FastAPI server'''
    # vaex_config_file: str = "vaex-server.json"
    add_example: bool = Field(True, title="Add example dataset")
    add_examples: bool = Field(True, title="Add example datasets")
    # vaex_config: dict = None
    graphql: bool = Field(False, title="Add graphql endpoint")
    sql: bool = Field(False, title="Add SQL endpoint")
    files: Dict[str, str] = Field(default_factory=list, title="Mapping of name to path")
    class Config(vaex.config.ConfigDefault):
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_prefix = 'vaex_server_'
        # secrets_dir = '/run/secrets'

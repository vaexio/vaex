import pytest
import vaex.server.fastapi
from fastapi.testclient import TestClient


@pytest.fixture(scope='session')
def request_client(webserver):
    client = TestClient(vaex.server.fastapi.app, raise_server_exceptions=True)
    return client

vaex.server.fastapi.ensure_example()

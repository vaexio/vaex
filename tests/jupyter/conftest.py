import pytest
import asyncio
import vaex.jupyter.utils


vaex.jupyter.utils._test_delay = 0.01

@pytest.fixture()
def flush_guard():
    assert not vaex.jupyter.utils._debounced_execute_queue, "oops, stuff was left in the queue"
    yield
    assert not vaex.jupyter.utils._debounced_execute_queue, "oops, stuff was left in the queue, please call flush before ending the test"


# as in https://github.com/erdewit/nest_asyncio/issues/20
@pytest.fixture(scope="session")
def event_loop():
    """Don't close event loop at the end of every function decorated by
    @pytest.mark.asyncio
    """
    import asyncio
    print("CREATE" * 10)
    # asyncio.set_event_loop()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    # vaex.jupyter.utils.main_event_loop = loop
    yield loop
    loop.close()


@pytest.fixture()
def server_latency(webserver):
    try:
        webserver._test_latency = 0.1
        yield
    finally:
        webserver._test_latency = None

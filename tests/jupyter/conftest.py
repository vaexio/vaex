import pytest
import asyncio
import vaex.jupyter.utils


vaex.jupyter.utils._test_delay = 0.01

@pytest.fixture()
def flush_guard():
    assert not vaex.jupyter.utils._debounced_execute_queue, "oops, stuff was left in the queue"
    yield
    assert not vaex.jupyter.utils._debounced_execute_queue, "oops, stuff was left in the queue, please call flush before ending the test"


# kept for pytest-asyncio <1; 1.x provides its own loop management and ignores this fixture
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


@pytest.fixture(scope="session")
def _shared_event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(autouse=True)
def _current_event_loop(_shared_event_loop):
    """Make sure there is a (shared) current event loop for the synchronous tests.

    Futures created by vaex.jupyter (e.g. in the debounce logic) need an event loop. Since cpython 3.14
    asyncio.get_event_loop() (and thus asyncio.Future()) no longer creates one implicitly, and
    pytest-asyncio>=1 ignores the event_loop fixture above.
    """
    asyncio.set_event_loop(_shared_event_loop)
    yield
    asyncio.set_event_loop(None)


@pytest.fixture()
def server_latency(webserver):
    try:
        webserver._test_latency = 0.1
        yield
    finally:
        webserver._test_latency = None

import asyncio
import threading

import vaex.asyncio


async def _wait():
    async with asyncio.timeout(1):
        for _ in range(3):
            await asyncio.sleep(0.001)
    return "done"


def _run_in_thread(fn):
    box = []
    thread = threading.Thread(target=lambda: box.append(fn()))
    thread.start()
    thread.join()
    (result,) = box
    if isinstance(result, BaseException):
        raise result
    return result


def test_with_event_loop_restores_previous_loop():
    loop_outer = asyncio.new_event_loop()
    asyncio.set_event_loop(loop_outer)
    with vaex.asyncio.with_event_loop():
        loop_inner = asyncio.get_event_loop()
    assert loop_inner is not loop_outer
    assert asyncio.get_event_loop() is loop_outer


def test_with_event_loop_without_prior_loop():
    # on cpython >= 3.14 (and on any non-main thread >= 3.10) there is no
    # implicitly created event loop, so with_event_loop must tolerate that;
    # run in a worker thread so the fallback branch is exercised everywhere
    def scenario():
        with vaex.asyncio.with_event_loop():
            inner = asyncio.get_event_loop()
        # read policy state without calling get_event_loop: once
        # nest_asyncio.apply() has run, a get_event_loop call silently
        # (re)creates and caches a loop instead of raising, so the original
        # 'no loop' state only shows in the policy's thread-local
        policy_loop = asyncio.get_event_loop_policy()._local._loop
        return inner, policy_loop

    inner, policy_loop = _run_in_thread(scenario)
    assert inner is not None
    if asyncio.get_event_loop_policy().get_event_loop.__module__.startswith("nest_asyncio"):
        # a nest-patched policy auto-creates and THREAD-locally caches a loop on
        # the first get_event_loop, so the worker thread never passes through a
        # 'no loop set' state; nothing assertable about the restoration there
        pass
    else:
        assert policy_loop is None


def test_just_run_nests_over_drawn_loop():
    # exercises the nest_asyncio-application path with asyncio.timeout() in the
    # coroutine: nest_asyncio 1.6.0 raises 'Timeout should be used inside a
    # task' there on cpython >= 3.14, nest-asyncio2 1.7.4 does not
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    assert vaex.asyncio.just_run(_wait()) == "done"


def test_just_run_without_loop():
    # no-loop path: just_run creates its own loop and keeps it set afterwards
    # (see the NOTE on the finally block in vaex/asyncio.py:just_run)
    def scenario():
        return vaex.asyncio.just_run(_wait())

    assert _run_in_thread(scenario) == "done"


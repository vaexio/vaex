import asyncio
import vaex.jupyter
import time
import pytest


@pytest.mark.asyncio
async def test_await_promise(df_trimmed):
    df = df_trimmed
    execute_time = 0
    @vaex.jupyter.debounced(delay_seconds=0.01)
    async def execute():
        nonlocal execute_time
        print("EXECUTE!!")
        execute_time = time.time()
        await df.execute_async()

    assert vaex.jupyter.utils.get_ioloop() is not None
    count_promise = df.count(df.x, delay=True)
    time_before_execute = time.time()
    execute()
    count = await count_promise
    assert execute_time > time_before_execute
    assert count == df.count(df.x)


def test_debounce_method(df_trimmed):
    class Foo:
        def __init__(self, count):
            self.count = count

        @vaex.jupyter.debounced(delay_seconds=0.01)
        def run(self):
            self.count += 1
            return self.count

    async def run():
        a = Foo(1)
        b = Foo(10)
        af = a.run()
        bf = b.run()
        assert a.run.obj is a
        assert b.run.obj is b
        assert af is not bf
        assert await af == 2
        assert await bf == 11

    asyncio.run(run())


@pytest.mark.parametrize("reentrant", [False, True])
def test_debounced_reentrant(reentrant):
    value = 0
    @vaex.jupyter.debounced(delay_seconds=0.01, reentrant=reentrant)
    async def execute():
        nonlocal value

        local_value = value
        await asyncio.sleep(0.02)
        # if reentrant (and invoked withing 0.02 seconds) both calls reference to 0, and return 0
        # if non-reentrant, the next call will always see the new value
        value = local_value + 1
        return {value}

    async def run():
        fa = execute()
        fb = execute()
        # we wait beyond the debounce time
        # so that the function is still executing (sleep)
        await asyncio.sleep(0.015)
        # but already call the next
        fc = execute()
        a = await fa
        b = await fb
        c = await fc
        if reentrant:
            assert a is b
            assert a == {1}
            assert c.issubset({1, 2}) # c can actually be called at the same time
            assert a is not c
        else:
            assert a is b
            assert a == {1}
            assert c == {2}
            assert a is not c

    asyncio.run(run())


def test_debounced_non_reentrant_hammer():
    running = 0
    @vaex.jupyter.debounced(delay_seconds=0.001, reentrant=False)
    async def execute():
        nonlocal running
        assert not running
        running = True
        await asyncio.sleep(0.001)
        running = False
        raise "bla"

    async def run():
        for i in range(10000):
            execute()
            await asyncio.sleep(0.001/4)

    asyncio.run(run())

def test_debounced_long_lasting():
    calls = 0
    @vaex.jupyter.debounced(delay_seconds=0.01)
    async def execute():
        nonlocal calls
        await asyncio.sleep(0.05)
        calls += 1
        return {calls}

    async def run():
        fa = execute()
        fb = execute()
        # we wait beyond the debounce time
        # so that the function is still executing (sleep)
        await asyncio.sleep(0.02)
        # but already call the next
        fc = execute()
        a = await fa
        b = await fb
        c = await fc
        assert a is b
        assert a == {1}
        assert c == {2}
        assert fa is fb
        assert a is not c

    asyncio.run(run())


@pytest.mark.parametrize("as_coroutine", [False, True])
@pytest.mark.parametrize("as_method", [False, True])
def test_debounced_await(df_trimmed, as_coroutine, as_method, flush_guard,):
    calls = 0

    if as_method:
        class Foo:
            if as_coroutine:
                @vaex.jupyter.debounced(delay_seconds=0.01)
                async def foo(self):
                    nonlocal calls
                    calls += 1
                    return {'calls': calls}

                @vaex.jupyter.debounced(delay_seconds=0.01)
                async def foo_error(self):
                    nonlocal calls
                    calls += 1
                    raise RuntimeError("foo")
            else:
                @vaex.jupyter.debounced(delay_seconds=0.01)
                def foo(self):
                    nonlocal calls
                    calls += 1
                    return {'calls': calls}

                @vaex.jupyter.debounced(delay_seconds=0.01)
                def foo_error(self):
                    nonlocal calls
                    calls += 1
                    raise RuntimeError("foo")
        foo2 = Foo()  # noqa
        foo1 = Foo()
        foo = foo1.foo
        foo_error = foo1.foo_error
        other_foo = foo2
    else:
        if as_coroutine:
            @vaex.jupyter.debounced(delay_seconds=0.01)
            async def foo():
                nonlocal calls
                calls += 1
                return {'calls': calls}

            @vaex.jupyter.debounced(delay_seconds=0.01)
            async def foo_error():
                nonlocal calls
                calls += 1
                raise RuntimeError("foo")
        else:
            @vaex.jupyter.debounced(delay_seconds=0.01)
            def foo():
                nonlocal calls
                calls += 1
                return {'calls': calls}

            @vaex.jupyter.debounced(delay_seconds=0.01)
            def foo_error():
                nonlocal calls
                calls += 1
                raise RuntimeError("foo")

    async def run():
        nonlocal calls
        assert calls == 0
        if as_method:
            calls -= 1  # we're gonna call it twice, so we correct
        future1 = foo()
        future2 = foo()
        if as_method:
            bla1 = other_foo.foo()
            bla2 = other_foo.foo()
        result1 = await future1
        result2 = await future2
        if as_method:
            await bla1
            await bla2
        assert calls == 1
        assert result1 is result2
        # await asyncio.sleep(0.02)

        if as_method:
            await bla1
            await bla2
            calls = 1

        future1b = foo()
        future2b = foo()
        result1b = await future1b
        result2b = await future2b
        assert calls == 2
        assert result1b is result2b
        assert result1 is not result1b

        future1 = foo_error()
        future2 = foo_error()
        with pytest.raises(RuntimeError) as e1:
            result1 = await future1
        assert str(e1.value) == 'foo'
        with pytest.raises(RuntimeError) as e2:
            result2 = await future2
        assert calls == 3
        assert e1.value is e2.value
        # await asyncio.sleep(0.02)

        future1b = foo_error()
        future2b = foo_error()
        with pytest.raises(RuntimeError) as e1b:
            result1b = await future1b
        # assert str(e1.value) == 'foo'
        # with pytest.raises(RuntimeError) as e2b:
        #     result2b = await future2b
        # assert calls == 4
        # assert e1b.value is e2b.value

        # assert e1.value is not e1b.value

    asyncio.run(run())

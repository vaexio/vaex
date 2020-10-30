def buffer(i, n=1):
    values = []
    try:
        for _ in range(n-1):
            values.append(next(i))
        while True:
            values.append(next(i))
            yield values.pop(0)
    except StopIteration:
        pass
    yield from values


def pmap(f, i, pool):
    for item in i:
        def call(args):
            return f(*args)
        yield pool.submit(call, item)


def pwait(i):
    for item in i:
        yield item.result()


def consume(i):
    for item in i:
        pass

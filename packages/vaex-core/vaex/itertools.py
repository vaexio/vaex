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


def filter_none(i):
    for item in i:
        if item is not None:
            yield item


def chunked(i, count):
    '''Yield list 'subslices' of iterator i with max length count.

    >>> list(chunked(range(10), 2))
    [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    >>> list(chunked(range(5), 2))
    [[0, 1], [2, 3], [4]]
    >>> list(chunked(range(3), 4))
    [[0, 1, 2]]
    >>> list(chunked(range(4), 4))
    [[0, 1, 2, 3]]
    '''
    values = []
    for el in i:
        values.append(el)
        if len(values) == count:
            yield values
            values = []
    if values:
        yield values

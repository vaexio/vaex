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

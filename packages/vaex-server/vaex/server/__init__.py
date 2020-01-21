def connect(url, **kwargs):
    """Connect to remote server (default is tornado)"""
    # dispatch to vaex.server.tornado package
    from .tornado_client import connect
    return connect(url, **kwargs)

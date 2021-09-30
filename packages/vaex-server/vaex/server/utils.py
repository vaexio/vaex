import json
import os
import logging

logger = logging.getLogger("vaex.webserver")


def exception(exception):
    logger.exception("handled exception at server, all fine: %r", exception)
    return ({"exception": {"class": str(exception.__class__.__name__), "msg": str(exception)}})


def error(msg):
    return ({"error": msg})


def get_overrides():
    return json.loads(os.environ.get('VAEX_SERVER_OVERRIDE', '{}'))

def hostname_override(hostname):
    overrides = get_overrides()
    if hostname in overrides:
        override = overrides[hostname]
        logger.warning('overriding hostname %s with %s', hostname, override)
        return override
    else:
        return hostname

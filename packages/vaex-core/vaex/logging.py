"""Sets up logging for vaex.

See `configuration of logging <conf.html#logging>`_ how to configure logging.

"""
import os
import vaex.settings
import logging

logger = logging.getLogger('vaex')
log_handler : logging.Handler = None


def set_log_level(loggers=["vaex"], level=logging.DEBUG):
    """set log level to debug"""
    for logger in loggers:
        logging.getLogger(logger).setLevel(level)


def set_log_level_debug(loggers=["vaex"]):
    """set log level to debug"""
    set_log_level(loggers, logging.DEBUG)


def set_log_level_info(loggers=["vaex"]):
    """set log level to info"""
    set_log_level(loggers, logging.INFO)


def set_log_level_warning(loggers=["vaex"]):
    """set log level to warning"""
    set_log_level(loggers, logging.WARNING)


def set_log_level_error(loggers=["vaex"]):
    """set log level to exception/error"""
    set_log_level(loggers, logging.ERROR)


def remove_handler():
    """Disabled logging, remove default hander and add null handler"""
    logging.getLogger('vaex').removeHandler(log_handler)
    logging.getLogger('vaex').addHandler(logging.NullHandler())

def reset():
    '''Reset configuration of logging (i.e. remove the default handler)'''
    logging.getLogger('vaex').removeHandler(log_handler)


def _set_log_level(conf, level):
    if conf:
        if conf.startswith('vaex'):
            set_log_level(conf.split(","), level=level)
        else:
            set_log_level(level=level)

def setup():
    """Setup logging based on the configuration in ``vaex.settings``

    This function is automatically called when importing vaex. If settings are changed, call :func:`reset` and this function again
    to re-apply the settings.
    """
    global log_handler

    if vaex.settings.main.logging.setup:
        logger.setLevel(logging.DEBUG)

        # create console handler and accept all loglevels
        if vaex.settings.main.logging.rich:
            from rich.logging import RichHandler
            log_handler = RichHandler()
        else:
            log_handler = logging.StreamHandler()

            # create formatter
            formatter = logging.Formatter('%(levelname)s:%(threadName)s:%(name)s:%(message)s')


            # add formatter to console handler
            log_handler.setFormatter(formatter)
        log_handler.setLevel(logging.DEBUG)

        # add console handler to logger
        logger.addHandler(log_handler)

    logging.getLogger("vaex").setLevel(logging.ERROR)  # default to higest level
    _set_log_level(vaex.settings.main.logging.error, logging.ERROR)
    _set_log_level(vaex.settings.main.logging.warning, logging.WARNING)
    _set_log_level(vaex.settings.main.logging.info, logging.INFO)
    _set_log_level(vaex.settings.main.logging.debug, logging.DEBUG)
    # VAEX_DEBUG behaves similar to VAEX_LOGGING_DEBUG, but has more effect
    DEBUG_MODE = os.environ.get('VAEX_DEBUG', '')
    if DEBUG_MODE:
        _set_log_level(DEBUG_MODE, logging.DEBUG)
    
setup()

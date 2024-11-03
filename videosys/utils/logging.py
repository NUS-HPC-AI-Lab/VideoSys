import logging


def init_logger():
    """
    Update the logger to write to a log file.
    """

    logger = logging.getLogger()
    logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

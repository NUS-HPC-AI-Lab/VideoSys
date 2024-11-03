import logging

import torch.distributed as dist


def init_logger(logging_dir: str = None, master_only: bool = True):
    """
    Update the logger to write to a log file.
    """
    if dist.is_initialized() and master_only:
        if_log = dist.get_rank() == 0
    else:
        if_log = True

    # clear existing logger
    logger = logging.getLogger()
    logger.handlers.clear()

    if if_log:
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            fmt="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s]   %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger()
    else:  # dummy logger
        logger = logging.getLogger()
        logger.addHandler(logging.NullHandler())

    return logger

import logging
import colorlog

def setup_logger():
    logger = logging.getLogger('image_search_engine')
    logger.setLevel(logging.DEBUG)  # Capture all levels

    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler with colored output
    handler = colorlog.StreamHandler()
    handler.setFormatter(
        colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    )
    logger.addHandler(handler)
    return logger

logger = setup_logger()
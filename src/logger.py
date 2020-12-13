import os
import logging

FILE = "game.log"
LOGGER = None
IS_KAGGLE = False
LEVEL = logging.DEBUG if not IS_KAGGLE else logging.INFO


class _FileHandler(logging.FileHandler):
    def emit(self, record):
        if IS_KAGGLE:
            print(self.format(record))
        else:
            super().emit(record)


def _get_logger():
    global LOGGER

    if not LOGGER:
        if not IS_KAGGLE:
            if os.path.exists(FILE):
                os.remove(FILE)

        LOGGER = logging.getLogger("394235ce-628f-4c68-abec-17b13d4b59f1")
        LOGGER.setLevel(LEVEL)
        ch = _FileHandler(FILE)
        ch.setLevel(LEVEL)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%H-%M-%S"
        )
        ch.setFormatter(formatter)
        LOGGER.addHandler(ch)

    return LOGGER


logger = _get_logger()

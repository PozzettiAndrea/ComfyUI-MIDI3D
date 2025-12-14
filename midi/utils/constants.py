import os
from contextlib import contextmanager

from .import_utils import is_flash3_available, is_sdpa_available

USE_FLASH3_BACKEND = is_flash3_available() and os.environ.get("USE_FLASH3", False)

USE_SDPA_BACKEND = is_sdpa_available() and os.environ.get("USE_SDPA", True)


@contextmanager
def disable_flash3():
    global USE_FLASH3_BACKEND
    old_value = USE_FLASH3_BACKEND
    USE_FLASH3_BACKEND = False
    yield
    USE_FLASH3_BACKEND = old_value

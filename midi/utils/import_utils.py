"""
Import utilities: Utilities related to imports and our lazy inits.
"""

import importlib
import importlib.util

import torch

_flash3_available = importlib.util.find_spec(
    "flash_attn_interface"
) is not None and "H800" in torch.cuda.get_device_name(0)
try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    _flash3_available = False

_is_sdpa_available = True
try:
    import torch.nn.functional as F

    _is_sdpa_available = hasattr(F, "scaled_dot_product_attention")
except ImportError:
    _is_sdpa_available = False


def is_flash3_available() -> bool:
    return _flash3_available


def is_sdpa_available() -> bool:
    return _is_sdpa_available

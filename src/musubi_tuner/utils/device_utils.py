from typing import Optional, Union
import os
import torch

# Opt-out of the per-call empty_cache (set NVFP4_NO_EMPTY_CACHE=1). empty_cache
# forces the allocator to release cached blocks to the driver, causing expensive
# re-allocations and device syncs when called frequently in the training loop.
_SKIP_EMPTY_CACHE = os.getenv("NVFP4_NO_EMPTY_CACHE", "0") == "1"


def clean_memory_on_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if _SKIP_EMPTY_CACHE:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "cpu":
        pass
    elif device.type == "mps":  # not tested
        torch.mps.empty_cache()


def synchronize_device(device: Optional[Union[str, torch.device]]):
    if device is None:
        return
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

from functools import wraps

import torch

__doc__ = """Device safety for torch functions.

The `ensure_device` decorator ensures that the default device is set to the
device of the first tensor argument of the decorated function. This is useful
for functions that create new tensors, such as `torch.randn`. Without this
decorator, the new tensor will be created on the CPU, even if the arguments
are on GPU.
"""


def ensure_device(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        default_device = torch.get_default_dtype()
        arg_devices = map(lambda x: getattr(x, "device", None), args)
        devices = filter(lambda x: x is not None, arg_devices)
        device = next(devices, None)
        try:
            torch.set_default_device(device)
            out = fn(*args, **kwargs)
        finally:
            torch.set_default_dtype(default_device)
        return out

    return wrapped_fn

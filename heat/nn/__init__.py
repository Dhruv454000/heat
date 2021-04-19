import sys
import torch
import unittest

from . import functional
from . import batch_norm


if sys.version_info.minor >= 7:
    from .data_parallel import *

    functional.__getattr__ = functional.func_getattr

    def __getattr__(name):
        torch_all = torch.nn.modules.__all__
        if name == "SyncBatchNorm":
            return batch_norm.HeatSyncBatchNorm
        elif name in torch_all:
            return torch.nn.__getattribute__(name)
        else:
            try:
                unittest.__getattribute__(name)
            except AttributeError:
                raise AttributeError(f"module {name} not implemented in Torch or Heat")


else:
    from . import data_parallel
    from . import tests

    class Wrapper(object):
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.torch_all = torch.nn.modules.__all__
            self.data_parallel_all = data_parallel.__all__

        def __getattr__(self, name):
            if name == "SyncBatchNorm":
                return batch_norm.HeatSyncBatchNorm
            elif name in self.torch_all:
                return torch.nn.__getattribute__(name)
            elif name == "functional":
                return functional
            elif name in self.data_parallel_all:
                return data_parallel.__getattribute__(name)
            elif name == "tests":
                return tests
            else:
                try:
                    unittest.__getattribute__(name)
                except AttributeError:
                    raise AttributeError(f"module '{name}' not implemented in Torch or Heat")

    sys.modules[__name__] = Wrapper(sys.modules[__name__])

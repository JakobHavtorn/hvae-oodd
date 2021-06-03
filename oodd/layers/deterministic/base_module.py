from typing import *

import torch
import torch.nn as nn

import oodd.layers.deterministic


def DeterministicModuleConstructor(config: Dict[str, Any], *args, **kwargs):
    """Construct the `DeterministicModule` given by the `'block'` argument in `config`

    NOTE Must not modify `config`
    """

    block_name = config["block"]
    config_kwargs = {k: v for k, v in config.items() if k != "block"}

    DeterministicModule = get_deterministic(block_name)

    module = DeterministicModule(*args, **kwargs, **config_kwargs)

    assert isinstance(module, DeterministicModule)
    return module


def get_deterministic(name: str):
    """Return the 'DeterministicModule' class corresponding to `name` if available"""
    try:
        klass = getattr(oodd.layers.deterministic, name)
    except KeyError:
        raise KeyError(f"No DeterministicModule of name {name} is defined.")
    return klass


class DeterministicModule(nn.Module):
    """Abstract base class for deterministic modules of stages

    Deterministic modules must implement a forward method that:
    - outputs a single `torch.Tensor`, potentially strided or otherwise up/down sampled.
    - takes a main input `x` which is given pre-activation if `residual==True` otherwise post-activation.
    - takes an auxilliary input `aux` (pre-activation) which will be a skip connection from one of the previous stages
      as defined by the output of this (or another `DeterministicModule`'s) forward method.

    See also `DeterministicModules`
    """

    def __init__(self, in_shape: Tuple, transposed: bool, residual: bool, aux_shape: Optional[Tuple] = None):

        super().__init__()

        self._in_shape = in_shape
        self.transposed = transposed
        self.residual = residual
        self.aux_shape = aux_shape

    @property
    def in_shape(self) -> Tuple:
        return self._in_shape

    @property
    def out_shape(self) -> Tuple:
        return self._out_shape

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError()

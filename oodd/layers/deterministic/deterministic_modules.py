from typing import Tuple, List, Dict, Optional

import numpy as np
import torch.nn as nn

from torch import Tensor

from .base_module import DeterministicModuleConstructor
from ..linear import NormedDense


class DeterministicModules(nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int],
        deterministic_configs: List[Dict[str, Tuple[int]]],
        in_residual: bool = True,
        transposed: bool = False,
        aux_shape: Optional[List[Tuple[int]]] = None,
        **kwargs
    ):
        """
        Defines a of sequence of deterministic modules (e.g. ResNets) with potential skip connections ('aux').

        You can extend this class by passing other `block`` classes as a part of the `deterministic_configs`.

        If the number of auxiliary inputs is smaller than the number of layers,
        the auxiliary inputs are repeated to match the number of layers.

        :param input_shape: input tensor shape as a tuple of integers (B, H, *D)
        :param deterministic_configs: describes the sequence of modules, each of them defined by a tuple  (filters, kernel_size, stride)
        :param in_residual: whether the first `DeterministicModule` has a residual input connection
        :param transposed: use transposed deterministic_configs
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param **kwargs: additional arguments to be passed to the `DeterministicModule`.
        """
        super().__init__()
        self.in_shape = input_shape
        self._use_skips = False if aux_shape is None else True

        layers = []
        if aux_shape is None:
            aux_shape = []

        for j, config in enumerate(deterministic_configs):
            residual = True if j > 0 else in_residual
            aux = aux_shape.pop() if self._use_skips else None

            block = DeterministicModuleConstructor(
                config=config, in_shape=input_shape, aux_shape=aux, transposed=transposed, residual=residual, **kwargs
            )

            input_shape = block.out_shape
            aux_shape = [input_shape] + aux_shape

            layers += [block]

        self.layers = nn.ModuleList(layers)
        self.out_shape = input_shape
        self.hidden_shapes = aux_shape

    def forward(self, x: Tensor, aux: Optional[List[Tensor]] = None, **kwargs) -> Tuple[Tensor, List[Tensor]]:
        """
        Pass input through each of the residual networks passing auxilliary inputs (if any) into each one.

        Auxilliary inputs flow in one of two ways:

        1. If a list of auxilliary inputs is passed, they will be consumed from the last element and downwards.
           Each of these inputs is then passed to the corresponding residual block.

        2. If a single auxilliary input is given, that input is fed to the first residual block. The output of that
           block is then fed as auxilliary input to the next one, and so forth.
           This behaviour takes over if a given list of auxilliary inputs is shorter than the number of layers.

        To understand better try to run the below:
            aux = [1, 2, 3, 4]
            for i in range(10):
                a = aux.pop()
                x = a ** 2  # Layer
                aux = [x] + aux
                print(i, a, aux)

        :param x: input tensor
        :param aux: list of auxiliary inputs
        :return: output tensor, activations
        """
        if aux is None:
            aux = []

        for layer in self.layers:
            a = aux.pop() if self._use_skips else None
            x = layer(x, a, **kwargs)
            aux = [x] + aux

        return x, aux

    def __len__(self):
        return len(self.layers)


class AsFeatureMap(nn.Module):
    def __init__(self, in_shape, target_shape, weightnorm=True, **kwargs):
        """Layer that converts a  input to match a target shape via a dense layer or identity.

        Args:
            in_shape (tuple): shape of the input tensor
            target_shape (tuple): shape of the output tensor
            weightnorm (bool, optional): Whether to use weight normalization on the dense transform. Defaults to True.
        """
        super().__init__()

        self._in_shape = in_shape

        if len(in_shape) < len(target_shape):
            out_features = np.prod(target_shape)
            self.transform = NormedDense(in_shape, out_features, weightnorm=weightnorm)
            self._out_shape = target_shape
        else:
            self.transform = None
            self._out_shape = in_shape

    def forward(self, x):
        if self.transform is None:
            return x

        x = self.transform(x)
        return x.view((-1, *self.out_shape))

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shape(self):
        return self._out_shape

from typing import *

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_module import DeterministicModule
from ..convolutions import SameConv2dWrapper, TransposeableNormedSameConv2d


class ResBlockConv2d(DeterministicModule):
    def __init__(
        self,
        in_shape: Tuple,
        kernel_size: int,
        out_channels: int = None,
        stride: int = 1,
        aux_shape: Optional[Tuple] = None,
        downsampling_mode: str = "convolutional",
        upsampling_mode: str = "convolutional",
        transposed: bool = False,
        residual: bool = True,
        weightnorm: bool = True,
        gated: bool = True,
        activation: nn.Module = nn.ReLU,
        dropout: Optional[float] = None,
    ):
        """A Gated Residual Network with stride and transposition, auxilliary input merging, weightnorm and dropout.

        Args:
            in_shape (tuple): input tensor shape (B x C x *D)
            out_channels (int): number of out_channels in convolution output
            kernel_size (int): size of convolution kernel
            stride (int): size of the convolution stride
            aux_shape (tuple): auxiliary input tensor shape (B x C x *D). None means no auxialiary input
            transposed (bool): transposed or not
            residual (bool): use residual connections
            weightnorm (bool): use weight normalization
            activation (nn.Module): activation function class
            dropout (float): dropout value. None is no dropout
        """
        super().__init__(in_shape=in_shape, transposed=transposed, residual=residual, aux_shape=aux_shape)

        # some parameters
        self.channels_in = in_shape[0]
        self.channels_out = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.resample_mode = upsampling_mode if transposed else downsampling_mode
        self.transposed = transposed
        self.residual = residual
        self.gated = gated
        self.activation_pre = activation() if self.residual else None

        # first convolution is always non-transposed and stride 1
        self.conv1 = TransposeableNormedSameConv2d(
            in_shape=in_shape,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            transposed=False,
            resample_mode="convolutional",
            weightnorm=weightnorm,
        )

        # aux op
        if aux_shape is not None:
            self.activation_aux = activation()

            if list(aux_shape[1:]) > list(self.conv1.out_shape[1:]):
                # Downsample height and width (and match channels)
                aux_stride = tuple(np.asarray(aux_shape[1:]) // np.asarray(self.conv1.out_shape[1:]))
                self.aux_op = TransposeableNormedSameConv2d(
                    in_shape=aux_shape,
                    out_channels=self.conv1.out_shape[0],
                    kernel_size=kernel_size,
                    stride=aux_stride,
                    transposed=False,
                    resample_mode=self.resample_mode,
                    weightnorm=weightnorm,
                )
            elif list(aux_shape[1:]) < list(self.conv1.out_shape[1:]):
                # Upsample height and width (and match channels)
                aux_stride = tuple(np.asarray(self.conv1.out_shape[1:]) // np.asarray(aux_shape[1:]))
                self.aux_op = TransposeableNormedSameConv2d(
                    in_shape=aux_shape,
                    out_channels=self.conv1.out_shape[0],
                    kernel_size=kernel_size,
                    stride=aux_stride,
                    transposed=True,
                    resample_mode=self.resample_mode,
                    weightnorm=weightnorm,
                )
            elif aux_shape[0] != self.conv1.out_shape[0]:
                # Change only channels using 1x1 convolution
                self.aux_op = TransposeableNormedSameConv2d(
                    in_shape=aux_shape,
                    out_channels=self.conv1.out_shape[0],
                    kernel_size=1,
                    stride=1,
                    transposed=False,
                    resample_mode=self.resample_mode,
                    weightnorm=weightnorm,
                )
            else:
                # aux_shape and out_shape are the same
                assert aux_shape == self.conv1.out_shape
                self.aux_op = None
        else:
            self.aux_op = None

        self.activation_mid = activation()

        # dropout
        self.dropout = nn.Dropout(dropout) if dropout else dropout

        # second convolution is potentially transposed and potentially resampling
        gated_channels = 2 * out_channels if self.gated else out_channels
        self.conv2 = TransposeableNormedSameConv2d(
            in_shape=self.conv1.out_shape,
            out_channels=gated_channels,
            kernel_size=kernel_size,
            stride=self.stride,
            weightnorm=weightnorm,
            transposed=transposed,
            resample_mode=self.resample_mode,
        )  # doubled out channels for gating

        # output shape
        self._out_shape = (out_channels, *self.conv2.out_shape[1:])  # always out_channels regardless of gating

        # residual connections
        self.residual_op = ResidualConnectionConv2d(self._in_shape, self._out_shape, residual)

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None, **kwargs: Any) -> torch.Tensor:
        # input activation: x = activation(x)
        x_act = self.activation_pre(x) if self.residual else x

        # conv 1: y = conv(x)
        y = self.conv1(x_act)

        # merge aux with x: y = y + f(aux)
        y = y + self.aux_op(self.activation_aux(aux)) if self.aux_op is not None else y

        # y = activation(y)
        y = self.activation_mid(y)

        # dropout
        y = self.dropout(y) if self.dropout else y

        # conv 2: y = conv(y)
        y = self.conv2(y)

        # gate: y = y_1 * sigmoid(y_2)
        if self.gated:
            h_stack1, h_stack2 = y.chunk(2, 1)
            sigmoid_out = torch.sigmoid(h_stack2)
            y = h_stack1 * sigmoid_out

        # resiudal connection: y = y + x
        y = self.residual_op(y, x)

        return y


class ResidualConnectionConv2d(nn.Module):
    """
    Handles residual connections for tensors with different shapes.
    Apply padding and/or avg pooling to the input when necessary
    """

    def __init__(self, in_shape, out_shape, residual=True):
        """
        args:
            in_shape (tuple): input module shape x
            out_shape (tuple): output module shape y=f(x)
            residual (bool): apply residual conenction y' = y+x = f(x)+x
        """
        super().__init__()
        self.residual = residual
        self.in_shape = in_shape
        self.out_shape = out_shape
        is_1d = len(in_shape) == 2

        # residual: channels
        if residual and self.out_shape[0] < self.in_shape[0]:
            # More channels in input than output: Simply remove as many as needed
            pad = int(self.out_shape[0]) - int(self.in_shape[0])
            self.residual_padding = [0, 0, 0, pad] if is_1d else [0, 0, 0, 0, 0, pad]

        elif residual and self.out_shape[0] > self.in_shape[0]:
            # Fewer channels in the input than output: Padd zero channels onto input
            pad = int(self.out_shape[0]) - int(self.in_shape[0])
            self.residual_padding = [0, 0, 0, pad] if is_1d else [0, 0, 0, 0, 0, pad]
            # warnings.warn(
            #     "The input has fewer feature maps than the output. "
            #     "There will be no residual connection for this layer: "
            #     f"{in_shape=}, {out_shape=}"
            # )
            # self.residual = False

        else:
            self.residual_padding = None

        # residual: height and width
        if residual and list(out_shape)[1:] < list(in_shape)[1:]:
            # Smaller hieight/width in output than input
            pool_obj = nn.AvgPool1d if len(out_shape[1:]) == 1 else nn.AvgPool2d
            stride = tuple((np.asarray(in_shape)[1:] // np.asarray(out_shape)[1:]).tolist())
            self.residual_op = SameConv2dWrapper(in_shape, pool_obj(3, stride=stride))

        elif residual and list(out_shape)[1:] > list(in_shape)[1:]:
            # Larger height/width in output than input
            # warnings.warn(
            #     "The height and width of the output are larger than the input. "
            #     "There will be no residual connection for this layer: "
            #     f"{in_shape=}, {out_shape=}"
            # )
            self.residual_op = nn.Upsample(size=self.out_shape[1:], mode="nearest")
            self.residual = False
        else:
            self.residual_op = None

    def forward(self, y, x):
        if not self.residual:
            return y

        x = F.pad(x, self.residual_padding) if self.residual_padding is not None else x
        x = self.residual_op(x) if self.residual_op is not None else x
        return y + x

    def __repr__(self):
        residual = self.residual
        residual_padding = self.residual_padding
        return f"ResidualConnectionConv2d({residual=}, {residual_padding=})"

from itertools import repeat
from collections import abc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _ntuple(n):
    """Given an integer, return that integer in an n-tuple. Given an Iterable, return that directly instead"""

    def parse(x):
        """Given an integer, return that integer in an n-tuple. Given an Iterable, return that directly instead"""
        if isinstance(x, abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def data_dependent_init_for_conv(convolution, x, init_scale=None, eps=1e-8):
    """Given a convolutional layer, initialize parameters dependent on data as described in [1]. Operates in-place.

    Like batch normalization, this method ensures that all features initially have zero mean and unit variance
    before application of the nonlinearity.

    [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
        https://arxiv.org/pdf/1602.07868.pdf

    Returns:
        torch.Tensor: The input after transformation by the initialized layer.
                      We transform before initialization and then correct the output to avoid two forward passes.
    """
    # initial values
    nn.init.kaiming_normal_(convolution._parameters["weight_v"].data, mode="fan_in", nonlinearity="relu")
    convolution._parameters["weight_g"].data.fill_(1.0)
    convolution._parameters["bias"].data.fill_(0.0)

    # get outout when uninitialized
    x = convolution(x)

    # compute normalization statistics
    all_dims_but_channel = (0, *list(range(x.ndim))[2:])  # Assumes channels first
    m_init, v_init = torch.mean(x, all_dims_but_channel), torch.var(x, all_dims_but_channel)
    scale_init = 1 / torch.sqrt(v_init + eps)

    # rescale weights according to [1]
    if convolution.transposed:
        scale_init_kernel_shaped = scale_init[None, :].view(convolution._parameters["weight_g"].size())
    else:
        scale_init_kernel_shaped = scale_init[:, None].view(convolution._parameters["weight_g"].size())

    convolution._parameters["weight_g"].data = scale_init_kernel_shaped
    convolution._parameters["bias"].data = -m_init * scale_init

    # return normalized output of layer
    if len(x.shape) == 3:  # 1D
        return scale_init[None, :, None] * (x - m_init[None, :, None])
    elif len(x.shape) == 4:  # 2D
        return scale_init[None, :, None, None] * (x - m_init[None, :, None, None])
    else:
        raise NotImplementedError("Not implemented but I think it is just the above with one more None")


def get_same_padding(in_shape, convolution):
    """
    Return the padding to apply to a given convolution such as it reproduces the 'same' behavior from Tensorflow

    This also works for pooling layers.

    Args:
        in_shape (tuple): Input tensor shape (B x C x D)
        convolution (nn.Module): Convolution module object
    returns:
        sym_padding, unsym_padding: Symmetric and unsymmetric padding to apply. We split in two because nn.Conv only
                                    allows setting symmetric padding so unsymmetric has to be done manually.
    """
    in_shape = np.asarray(in_shape)[1:]
    kernel_size = np.asarray(convolution.kernel_size)
    pad = np.asarray(convolution.padding)
    dilation = np.asarray(convolution.dilation) if hasattr(convolution, "dilation") else 1
    stride = np.asarray(convolution.stride)

    # handle pooling layers
    if not hasattr(convolution, "transposed"):
        convolution.transposed = False
    else:
        assert len(in_shape) == len(kernel_size), "tensor is not the same dimension as the kernel"

    if not convolution.transposed:
        effective_filter_size = (kernel_size - 1) * dilation + 1
        output_size = (in_shape + stride - 1) // stride
        padding_input = np.maximum(0, (output_size - 1) * stride + (kernel_size - 1) * dilation + 1 - in_shape)
        odd_padding = padding_input % 2 != 0
        sym_padding = tuple(padding_input // 2)
        unsym_padding = [y for x in odd_padding for y in [0, int(x)]]
    else:
        padding_input = kernel_size - stride
        sym_padding = None
        unsym_padding = [
            y for x in padding_input for y in [-int(np.floor(int(x) / 2)), -int(np.floor(int(x) / 2) + int(x) % 2)]
        ]

    return sym_padding, unsym_padding


def get_out_shape_same_padding(in_shape, out_channels, stride, transposed):
    """Compute output shape for a same padded (potentially transposed) convolution with given channels and stride"""
    stride_factor = np.asarray(stride)
    out_shape = np.asarray(in_shape)
    if transposed:
        out_shape[1:] = out_shape[1:] * stride_factor
    else:
        out_shape[1:] = out_shape[1:] // stride_factor
    out_shape[0] = out_channels
    out_shape = tuple(out_shape.astype(int))
    return out_shape


class SameConv2dWrapper(nn.Module):
    """Wraps a convolution layer instance with `same` padding (as in tensorflow).

    Works by computing the amount of same padding required, splitting this into symmetric and unsymmetric padding.
    The symmetric padding is added directly to the convolution object while the unsymmetric padding, which cannot, is
    padded explicitly during the forward pass.

    For transposed convolutions, the unsymmetric padding is done after the convolution - for regular convolutions it
    is done before.

    This class requires providing the complete input tensor shape (not just channels).
    """

    def __init__(self, in_shape, conv):
        """
        Args:
            in_shape (tuple): input tensor shape
            conv (nn.Module): convolution instance
        """
        super().__init__()
        self._in_shape = in_shape
        self.conv = conv

        # Get padding
        sym_padding, self.unsym_padding = get_same_padding(in_shape, self.conv)
        if not self.conv.transposed:
            self.conv.padding = sym_padding

        transposed = hasattr(conv, "transposed") and conv.transposed
        out_channels = conv.out_channels if hasattr(conv, "out_channels") else in_shape[0]
        self._out_shape = get_out_shape_same_padding(in_shape, out_channels, conv.stride, transposed)

    def forward(self, x):
        if not self.conv.transposed:
            x = F.pad(x, self.unsym_padding)

        x = self.conv(x)

        if self.conv.transposed:
            x = F.pad(x, self.unsym_padding)

        return x

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def out_shape(self):
        return self._out_shape


class NormedSameConv2dWrapper(SameConv2dWrapper):
    """Wraps a convolution layer instance with `same` padding (as in tensorflow) and weight normalization.

    This class also handles data dependent initialization for weight normalization.

    The same padding requires the full in_shape rather than just the input channels.
    """

    def __init__(self, in_shape, conv, weightnorm=True, init_scale=None):
        """
        args:
            in_shape (tuple): input tensor shape (B x C x D)
            conv (nn.Module): convolution instance of type Conv1d, ConvTranspose1d, Conv2d or ConvTranspose2d
            weightnorm (bool): use weight normalization
        """
        super().__init__(in_shape, conv)
        self.weightnorm = weightnorm
        self.init_scale = init_scale

        self.register_buffer("initialized", torch.tensor(False))

        if weightnorm:
            dim = 1 if self.conv.transposed else 0
            self.conv = nn.utils.weight_norm(self.conv, dim=dim, name="weight")

    def initialize(self, x, init_scale=None):
        self.initialized = True + self.initialized
        if self.weightnorm:
            data_dependent_init_for_conv(self.conv, x, init_scale)

    def forward(self, x):
        if not self.conv.transposed:
            x = F.pad(x, self.unsym_padding)

        if not self.initialized:
            self.initialize(x, self.init_scale)

        x = self.conv(x)

        if self.conv.transposed:
            x = F.pad(x, self.unsym_padding)

        return x


class TransposeableNormedSameConv2d(NormedSameConv2dWrapper):
    def __init__(
        self,
        in_shape,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        transposed: bool = False,
        resample_mode: str = "convolutional",
        weightnorm=True,
        init_scale=None,
    ):
        """A transposeable same-padded convolution with weight normalizeation.

        If `stride == 1`, the layer always creates a regular convolution. Hence, the `transpose` and `resample_mode`
        arguments only have an effect if `stride > 1`.

        Has the option to replace strided convolutions (tranposed or not) with nearest neighbour or bilinear interpolation
        followed by a stride 1 convolution. This can be used to get behaviour as in [1] or [2].

        [1] https://arxiv.org/abs/2011.10650
        [2] https://arxiv.org/pdf/2003.01826.pdf

        Args:
            in_shape ([type]): [description]
            out_channels ([type]): [description]
            kernel_size ([type]): [description]
            stride (int, optional): [description]. Defaults to 1.
            dilation (int, optional): [description]. Defaults to 1.
            groups (int, optional): [description]. Defaults to 1.
            bias (bool, optional): [description]. Defaults to True.
            padding_mode (str, optional): [description]. Defaults to "zeros".
            transposed (bool, optional): [description]. Defaults to False.
            resample_mode (str, optional): The method for resampling if the stride is non-unit. Defaults to "convolutional".
            weightnorm (bool, optional): [description]. Defaults to True.
            init_scale ([type], optional): [description]. Defaults to None.
        """

        assert resample_mode in ["convolutional", "bilinear", "nearest"], "resample_mode must be one of these"

        stride = _pair(stride)

        if stride > (1, 1) and resample_mode != "convolutional":
            scale_factor = stride if transposed else tuple(1 / np.asarray(stride))
            resample = nn.Upsample(scale_factor=scale_factor, mode=resample_mode)
            conv_stride = 1
        else:
            resample = None
            conv_stride = stride

        use_transposed_conv = stride > (1, 1) and transposed and resample_mode == "convolutional"
        if not use_transposed_conv:
            conv_obj = nn.Conv2d
        else:
            conv_obj = nn.ConvTranspose2d

        in_channels = in_shape[0]
        conv = conv_obj(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=conv_stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        super().__init__(in_shape, conv, weightnorm=weightnorm, init_scale=init_scale)

        self.resample = resample
        self._out_shape = get_out_shape_same_padding(in_shape, out_channels, stride, transposed)

    def forward(self, x):
        x = self.resample(x) if self.resample is not None else x

        if not self.conv.transposed:
            x = F.pad(x, self.unsym_padding)

        if not self.initialized:
            self.initialize(x, self.init_scale)

        x = self.conv(x)

        if self.conv.transposed:
            x = F.pad(x, self.unsym_padding)
        return x


class SameConv2d(SameConv2dWrapper):
    """A Conv2d modified to always have `same` padding.

    Requires full input shape instead of channels. Assumes channels first, i.e. (B, C, H, W)
    """

    def __init__(
        self, in_shape, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"
    ):
        in_channels = in_shape[0]
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        super().__init__(in_shape, conv)


class SameConvTranspose2d(SameConv2dWrapper):
    """A ConvTransposed2d modified to always have `same` padding.

    Requires full input shape instead of channels. Assumes channels first, i.e. (B, C, H, W)
    """

    def __init__(
        self, in_shape, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, padding_mode="zeros"
    ):
        in_channels = in_shape[0]
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        super().__init__(in_shape, conv)


class NormedSameConv2d(NormedSameConv2dWrapper):
    """A Conv2d modified to always have `same` padding and use weight normalization.

    Requires full input shape instead of channels. Assumes channels first, i.e. (B, C, H, W)
    """

    def __init__(
        self,
        in_shape,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        weightnorm=True,
    ):
        in_channels = in_shape[0]
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        super().__init__(in_shape, conv, weightnorm=weightnorm)


class NormedSameConvTranspose2d(NormedSameConv2dWrapper):
    """A ConvTransposed2d modified to always have `same` padding and use weight normalization.

    Requires full input shape instead of channels. Assumes channels first, i.e. (B, C, H, W)
    """

    def __init__(
        self,
        in_shape,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        weightnorm=True,
    ):
        in_channels = in_shape[0]
        conv = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        super().__init__(in_shape, conv, weightnorm=weightnorm)

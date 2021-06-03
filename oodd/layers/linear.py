from typing import Tuple, Union

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as D

from torch import Tensor


def data_dependent_init_for_linear(linear, x, init_scale=None, eps=1e-8):
    """Given a linear layer, initialize parameters dependent on data as described in [1]. Operates in-place.

    Like batch normalization, this method ensures that all features initially have zero mean and unit variance
    before application of the nonlinearity.

    [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
        https://arxiv.org/pdf/1602.07868.pdf

    Returns:
        torch.Tensor: The input after transformation by the initialized layer.
                      We transform before initialization and then correct the output to avoid two forward passes.
    """
    if x.ndim == 1 or not x.shape[0] > 1:
        raise RuntimeError(f"Cannnot do data-based weightnorm initialization without a batch {x.shape=}")

    # initial random values
    nn.init.kaiming_normal_(linear._parameters["weight_v"].data, mode='fan_in', nonlinearity='relu')
    linear._parameters["weight_g"].data.fill_(1.0)
    linear._parameters["bias"].data.fill_(0.0)

    # data dependent initialization
    x = linear(x)
    m_init, v_init = torch.mean(x, 0), torch.var(x, 0)

    scale_init = 1 / torch.sqrt(v_init + eps)
    scale_init_weight_shaped = scale_init.view(linear._parameters["weight_g"].data.size())

    linear._parameters["weight_g"].data = scale_init_weight_shaped
    linear._parameters["bias"].data = - m_init * scale_init
    return scale_init[None, :] * (x - m_init[None, :])


class NormedLinear(nn.Module):
    """Linear layer with weight normalization as in [1]"""

    def __init__(self, in_features, out_features, dim=-1, weightnorm=True, init_scale=None):
        super().__init__()
        """Linear layer with weight normalization [1]. Operates on `dim` and broadcasts to others.

        [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
            https://arxiv.org/pdf/1602.07868.pdf

        Args:
            in_features (in): number of input features
            out_features (int): number of output features
            dim (int): dimension to aply transformation to
            weightnorm (bool): use weight normalization
        """
        super().__init__()
        self._in_features = in_features
        self._out_features = out_features
        self.dim = dim
        self.weightnorm = weightnorm
        self.init_scale = init_scale

        self.register_buffer("initialized", torch.tensor(False))

        self.linear = nn.Linear(self._in_features, out_features)
        if weightnorm:
            self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")

    def initialize(self, x, init_scale=None):
        if self.weightnorm:
            data_dependent_init_for_linear(self.linear, x, init_scale)
            self.initialized = True + self.initialized

    def forward(self, x):
        # Reshape in according to which dimension the layer operates on
        shape_original = list(x.size())
        dim = self.dim if self.dim >= 0 else x.ndim + self.dim

        permute_dims = dim < x.ndim - 1  # Permute if not the last dimension
        if permute_dims:
            x = x.flatten(start_dim=dim + 1)  # Flatten all dimensions after 'dim' into a new rightmost dimension
            x = x.transpose(-1, -2).contiguous()  # Switch rightmost dimension and `dim` (so `dim` is rightmost)
            shape_permuted_in = list(x.shape)
            shape_permuted_out = shape_permuted_in[:-1] + [self._out_features]

        x = x.view(-1, x.size(-1))  # Collapse all dims except last into the batch dimension

        # Initialize and transform
        if not self.initialized:
            self.initialize(x, self.init_scale)

        x = self.linear(x)

        # Reshape out according to which dimension the layer operated on
        shape_original[dim] = self._out_features
        if permute_dims:
            x = x.view(shape_permuted_out).transpose(-1, -2)

        x = x.reshape(shape_original)
        return x

    @property
    def in_shape(self):
        return (self._in_features,)

    @property
    def out_shape(self):
        return (self._out_features,)


class NormedDense(nn.Module):
    """Dense layer with weight normalization as in [1]"""

    def __init__(
        self,
        in_shape: Union[int, Tuple[int]],
        out_features: int,
        bias: bool = True,
        weightnorm: bool = True,
        init_scale: float = None,
    ):
        """Densely connected layer with weight normalization [1]

        [1] Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks \
            https://arxiv.org/pdf/1602.07868.pdf

        Args:
            in_shape (tuple or int): input tensor shape (B x C x D)
            out_features (int): number of output features
            weightnorm (bool): use weight normalization
        """
        super().__init__()
        self._input_shape = in_shape if isinstance(in_shape, tuple) else (in_shape,)
        self.input_features = int(np.prod(in_shape))
        self._output_shape = (out_features,)
        self.bias = bias
        self.weightnorm = weightnorm
        self.init_scale = init_scale

        self.register_buffer("initialized", torch.tensor(False))

        self.linear = nn.Linear(self.input_features, out_features, bias=bias)

        if weightnorm:
            self.linear = nn.utils.weight_norm(self.linear, dim=0, name="weight")

    def initialize(self, x, init_scale=None):
        if self.weightnorm:
            data_dependent_init_for_linear(self.linear, x, init_scale)
            self.initialized = True + self.initialized

    def forward(self, x):
        x = x.flatten(start_dim=1)

        if not self.initialized:
            self.initialize(x, self.init_scale)

        x = self.linear(x)
        return x

    @property
    def in_shape(self):
        return self._input_shape

    @property
    def out_shape(self):
        return self._output_shape


class LinearVariational(nn.Module):
    """Mean field approximation of nn.Linear"""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        """
        Parameters
        ----------
        in_features : int
        out_features : int
        n_batches : int
            Needed for KL-divergence re-scaling.
            See Also:
                Blundell, Cornebise, Kavukcioglu & Wierstra (2015, May 20)
                Weight Uncertainty in Neural Networks.
                Retrieved from https://arxiv.org/abs/1505.05424
        bias
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.include_bias = bias
        self._kl_divergence_ = 0

        # Initialize the variational parameters (variance as log(1 + exp(ρ)) * eps)
        self.w_mu = nn.Parameter(torch.FloatTensor(in_features, out_features))  # .normal_(mean=0, std=0.001)
        self.w_p = nn.Parameter(torch.FloatTensor(in_features, out_features))  # .normal_(mean=-2.5, std=0.001)
        if bias:
            self.b_mu = nn.Parameter(torch.zeros(out_features), requires_grad=True)
            self.b_p = nn.Parameter(torch.zeros(out_features), requires_grad=True)
        else:
            self.register_parameter("b_mu", None)
            self.register_parameter("b_p", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.w_mu, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.b_mu, -bound, bound)

    def reparameterize(self, mu: Tensor, p: Tensor) -> Tensor:
        sigma = torch.log(1 + torch.exp(p))
        eps = torch.randn_like(sigma)
        return mu + (eps * sigma)

    def kl_divergence(
        self,
        z: Tensor,
        mu_theta: Tensor,
        p_theta: Tensor,
        prior_sd: float = 1.0,
    ) -> float:
        log_prior = D.Normal(0, prior_sd).log_prob(z)
        log_p_q = D.Normal(mu_theta, torch.log(1 + torch.exp(p_theta))).log_prob(z)
        return (log_p_q - log_prior).sum() / self.n_batches

    def forward(self, x: Tensor) -> Tensor:
        w = self.reparameterize(self.w_mu, self.w_p)
        b = self.reparameterize(self.b_mu, self.b_p) if self.include_bias else None

        z = F.linear(x, w, b)  # z = x @ w + b

        self._kl_divergence_ += self.kl_divergence(w, self.w_mu, self.w_p)
        if self.include_bias:
            self._kl_divergence_ += self.kl_divergence(b, self.b_mu, self.b_p)

        return z

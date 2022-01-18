"""
Stochastic layers are modules that define a stochastic transformation
usually using the reparameterization trick to provide differentiability.
"""

from collections import namedtuple
from typing import Dict, Tuple, Union, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torch import Tensor

import oodd.layers.stochastic

from oodd.utils import reduce_to_batch, reduce_to_latent
from oodd.variational import kl_divergence_mc

from .activations import InverseSoftplus
from .linear import NormedDense
from .convolutions import NormedSameConv2d


def StochasticModuleConstructor(config: Dict[str, Any], *args, **kwargs):
    """Construct the `StochasticModule` given by the `'block'` argument in `config`

    NOTE Must not modify `config`.
    """
    block_name = config["block"]
    config_kwargs = {k: v for k, v in config.items() if k != "block"}

    StochasticModule = get_stochastic(block_name)

    try:
        module = StochasticModule(*args, **kwargs, **config_kwargs)
    except TypeError as exc:
        raise TypeError(f"{str(exc)} ('{block_name}')")

    assert isinstance(module, StochasticModule)
    return module


def get_stochastic(name: str):
    """Return the 'StochasticModule' class corresponding to `name` if available"""
    try:
        klass = getattr(oodd.layers.stochastic, name)
    except KeyError:
        raise KeyError(f"No StochasticModule of name {name} is defined.")
    return klass


STOCHASTIC_DATA_FIELDS = ["z", "dist", "mean", "variance", "use_mode", "forced_latent"]
StochasticData = namedtuple(
    typename="StochasticData",
    field_names=STOCHASTIC_DATA_FIELDS,
    defaults=[None] * len(STOCHASTIC_DATA_FIELDS),
)

LOSS_DATA_FIELDS = [
    "loss",
    "p_logprob",
    "q_logprob",
    "kl_samplewise",
    "kl_latentwise",
    "kl_elementwise",
    "use_mode",
    "forced_latent",
]
LossData = namedtuple(
    typename="LossData", field_names=LOSS_DATA_FIELDS, defaults=[None] * len(LOSS_DATA_FIELDS)
)


class StochasticModule(nn.Module):
    """Abstract base class for stochastic layers"""

    def __init__(self, in_shape: Union[int, Tuple[int]], latent_features: int, top: bool):
        super().__init__()
        self._in_shape = in_shape if isinstance(in_shape, tuple) else (in_shape,)
        self.latent_features = latent_features
        self.top = top

    @property
    def out_shape(self):
        return self._out_shape

    @property
    def in_shape(self):
        return self._in_shape

    @property
    def prior(self):
        raise NotImplementedError()

    def infer(
        self,
        x: Tensor,
        sample: bool = True,
        n_prior_samples: Optional[int] = 1,
        forced_latent: Tensor = None,
        use_mode: bool = False,
    ) -> Tuple[Tensor, StochasticData]:
        return self(
            x=x,
            inference=True,
            sample=sample,
            n_prior_samples=n_prior_samples,
            forced_latent=forced_latent,
            use_mode=use_mode,
        )

    def generate(
        self,
        x: Optional[Tensor] = None,
        sample: bool = True,
        n_prior_samples: Optional[int] = 1,
        forced_latent: Tensor = None,
        use_mode: bool = False,
    ) -> Tuple[Tensor, StochasticData]:
        return self(
            x=x,
            inference=False,
            sample=sample,
            n_prior_samples=n_prior_samples,
            forced_latent=forced_latent,
            use_mode=use_mode,
        )

    def forward(
        self,
        x: Optional[Tensor],
        inference: bool,
        sample: bool = True,
        n_prior_samples: Optional[int] = 1,
        forced_latent: Tensor = None,
        use_mode: bool = False,
    ) -> Tuple[Tensor, StochasticData]:
        """
        Returns the distribution parametrized by the outputs of a transformation if x and sample if `sample`=True.
        If no hidden state is provided, sample from the prior.

        :param x: hidden state used to computed logits (Optional : None means using the prior)
        :param inference: inference mode switch
        :param sample: sample layer
        :param n_prior_samples: number of samples (when sampling from prior)
        :param kwargs: additional args passed ot the stochastic layer
        :return: (projected sample, data)
        """
        raise NotImplementedError

    def loss(self, q_data: StochasticData, p_data: StochasticData) -> LossData:
        """
        Compute the KL divergence and other auxiliary losses if required

        :param q_data: data received from the posterior forward pass
        :param p_data: data received from the prior forward pass
        :param kwargs: other parameters passed to the kl function
        :return: dictionary of losses {'kl': [values], 'auxiliary' : [aux_values], ...}
        """
        raise NotImplementedError

    def get_generative_parameters(self):
        raise NotImplementedError()

    def get_inference_parameters(self):
        raise NotImplementedError()


class GaussianStochasticModule(StochasticModule):
    """Base module for StochasticModules with a diagonal covariance Gaussian distribution.

    Subclasses determine how to parameterize the q and p distributions and hence must define:
    - self.in_transform_q
    - self.in_transform_p
    - self._out_shape
    """

    def __init__(
        self,
        in_shape: Union[int, Tuple[int]],
        latent_features: int,
        top: bool = False,
        activation: nn.Module = nn.ELU,
        learn_prior: bool = False,
        min_scale: float = 1e-6,
        max_scale: float = float("inf")
    ):
        super().__init__(in_shape, latent_features, top)

        self.activation = activation()
        self.learn_prior = learn_prior
        self.min_scale = min_scale
        self.max_scale = max_scale

        # Activation to get standard deviation from linear output.
        # beta = ln(2) results in softplus(0) = 1 which in turn gives a standard normal distribution at initialization
        self.std_activation = nn.Softplus(beta=np.log(2))
        self.std_activation_inverse = InverseSoftplus(beta=np.log(2))

        self.in_transform_q = None
        self.in_transform_p = None

    def define_prior(self, *shape):
        """Define the prior as standard normal

        Since we apply Softplus to the scale paramteer we must do the inverse for the initial value.
        """
        mu = torch.zeros(*shape)
        sigma = torch.ones(*shape)
        log_scale = self.std_activation_inverse(sigma)

        prior_logits = torch.cat([mu, log_scale])
        if self.learn_prior:
            self.prior_logits = nn.Parameter(prior_logits)
        else:
            self.register_buffer("prior_logits", prior_logits)

    @property
    def prior(self):
        """Return the prior distribution without a batch dimension"""
        mu, sigma = self.logits_to_mu_and_sigma(self.prior_logits, batched=False)
        return D.Normal(mu, sigma)

    def logits_to_mu_and_sigma(self, logits, batched=True):
        """Convert logits to parameters for the Normal. We chunk on axis 0 or 1 depending on batching"""
        mu, log_scale = logits.chunk(2, dim=int(batched))
        sigma = self.std_activation(log_scale)
        return mu, sigma

    def compute_params(self, x: Tensor, inference: bool) -> Tuple[Tensor, Tensor]:
        """
        Compute the logits of the distribution.
        :param x: input tensor
        :param inference: inference mode or not (generative mode)
        :return: logits
        """
        x = self.activation(x)

        if inference:
            logits = self.in_transform_q(x)
        else:
            logits = self.in_transform_p(x)

        mu, sigma = self.logits_to_mu_and_sigma(logits)
        return mu, sigma

    def forward(
        self,
        x: Optional[Tensor],
        inference: bool,
        sample: Optional[bool] = True,
        n_prior_samples: Optional[int] = 1,
        forced_latent: Optional[Tensor] = None,
        use_mode: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[Tensor, StochasticData]:

        # Define distribution
        if x is None:
            mu, sigma = self.logits_to_mu_and_sigma(self.prior_logits.expand(n_prior_samples, *self.prior_logits.shape))
        else:
            mu, sigma = self.compute_params(x, inference)

        sigma = torch.clamp(sigma, min=self.min_scale, max=self.max_scale)

        dist = D.Normal(mu, sigma)

        # Define latent variable
        if forced_latent is not None:
            if forced_latent.shape[0] != dist.batch_shape[0]:
                forced_latent = forced_latent.expand(dist.batch_shape)
            z = forced_latent
        elif use_mode:
            z = dist.mean
        elif sample:
            z = dist.rsample()
        else:
            z = None

        data = StochasticData(
            z=z,
            mean=dist.mean,
            variance=dist.variance,
            dist=dist,
            use_mode=torch.tensor(use_mode),
            forced_latent=torch.tensor(forced_latent is not None),
        )
        return z, data

    def loss(self, q_data: StochasticData, p_data: StochasticData) -> LossData:
        kl_elementwise, q_logprob, p_logprob = kl_divergence_mc(q_data.z, q_data.dist, p_data.dist)
        kl_latentwise = reduce_to_latent(kl_elementwise)
        kl_samplewise = reduce_to_batch(kl_latentwise)
        return LossData(
            q_logprob=q_logprob,
            p_logprob=p_logprob,
            loss=kl_samplewise,  # (B,)
            kl_samplewise=kl_samplewise,  # (B,)
            kl_latentwise=kl_latentwise,  # (B, L)
            kl_elementwise=kl_elementwise,  # (B, L, D*)
            use_mode=q_data.use_mode,
            forced_latent=q_data.forced_latent,
        )

    def get_generative_parameters(self):
        if self.in_transform_p is not None:
            yield from self.in_transform_p.parameters()
        yield from ()

    def get_inference_parameters(self):
        if self.in_transform_q is not None:
            yield from self.in_transform_q.parameters()
        yield from ()


class GaussianDense(GaussianStochasticModule):
    """A Normal stochastic layer parametrized by a dense layer."""

    def __init__(
        self,
        in_shape: Union[int, Tuple[int]],
        latent_features: int,
        top: bool = False,
        activation: nn.Module = nn.ELU,
        learn_prior: bool = False,
        weightnorm: bool = True,
    ):

        super().__init__(
            in_shape=in_shape, latent_features=latent_features, top=top, activation=activation, learn_prior=learn_prior
        )

        if top:
            self.define_prior(self.latent_features)

        # computes logits
        nz_in = 2 * self.latent_features
        self.in_transform_q = NormedDense(in_shape, nz_in, weightnorm=weightnorm)
        if not top:
            self.in_transform_p = NormedDense(in_shape, nz_in, weightnorm=weightnorm)

        self._out_shape = (self.latent_features,)


class GaussianConv2d(GaussianStochasticModule):
    """A Normal stochastic layer parametrized by a convolution."""

    def __init__(
        self,
        in_shape: Tuple[int],
        latent_features: int,
        top: bool = False,
        activation: nn.Module = nn.ELU,
        learn_prior: bool = False,
        kernel_size: Union[int, Tuple[int]] = 3,
        weightnorm: bool = True,
        **kwargs
    ):

        super().__init__(
            in_shape=in_shape, latent_features=latent_features, top=top, activation=activation, learn_prior=learn_prior
        )

        if top:
            self.define_prior(self.latent_features, *in_shape[1:])

        # computes logits
        nz_in = 2 * self.latent_features
        self.in_transform_q = NormedSameConv2d(
            in_shape, out_channels=nz_in, kernel_size=kernel_size, weightnorm=weightnorm, **kwargs
        )
        if not top:
            self.in_transform_p = NormedSameConv2d(
                in_shape, out_channels=nz_in, kernel_size=kernel_size, weightnorm=weightnorm, **kwargs
            )

        # compute output shape
        self._out_shape = (self.latent_features, *in_shape[1:])

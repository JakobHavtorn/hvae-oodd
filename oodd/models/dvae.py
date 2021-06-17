import argparse

from typing import *

import torch
import torch.nn as nn

from oodd.layers.likelihoods import LikelihoodModule, get_likelihood, LikelihoodData
from oodd.layers.stages import StageData, StageModule, VaeStage, LvaeStage, BivaStage
from oodd.layers.stochastic import StochasticData
from oodd.utils.argparsing import str2bool, json_file_or_json
from oodd.models import BaseModule


class DeepVAE(BaseModule):
    """
    A Deep Hierarchical VAE.

    The model is a stack of N stages. Each stage features an inference and a generative path.

    Depending on the choice of the stage, multiple models can be implemented:
    - VAE: https://arxiv.org/abs/1312.6114
    - LVAE: https://arxiv.org/abs/1602.02282
    - BIVA: https://arxiv.org/abs/1902.02102
    """

    def __init__(
        self,
        Stage: StageModule,
        input_shape: Tuple[int],
        likelihood_module: LikelihoodModule,
        config_deterministic: List[List[Dict[str, Any]]],
        config_stochastic: List[Dict[str, Any]],
        activation: str = "Swish",
        q_dropout: float = 0.0,
        p_dropout: float = 0.0,
        skip_stochastic: bool = True,
        padded_shape: Optional[Tuple] = None,
        features_out: Optional[int] = None,
        lambda_init: Optional[Callable] = None,
        **stage_kwargs,
    ):
        """
        Initialize the Deep VAE model.
        :param Stage: stage constructor (VaeStage, LvaeStage, BivaStage)
        :param input_shape: Input tensor shape (batch_size, channels, *dimensions)
        :param likelihood_module: likelihood_module module with constructor __init__(in_shape, out_features)
        :param config_deterministic: one list of key-value configs in a dict, for each deterministic module.
        :param config_stochastic: a list of key-value configs in a dict, each describing a stochastic module for a stage
        :param activation: activation function (e.g. gelu, elu, relu, tanh)
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param skip_stochastic: whether to have skip connections between stochastic latent variables
        :param padded_shape: pad input tensor to this shape for instance if downsampling many times
        :param features_out: optional number of output features if different from the input
        :param lambda_init: lambda function applied to the input
        :param stage_kwargs: additional arugments passed to each stage
        """
        super().__init__()

        assert len(config_deterministic) == len(config_stochastic)

        self.input_shape = input_shape
        self.likelihood_module = likelihood_module
        self.config_deterministic = config_deterministic
        self.config_stochastic = config_stochastic
        self.activation = activation
        self.q_dropout = q_dropout
        self.p_dropout = p_dropout
        self.skip_stochastic = skip_stochastic
        self.padded_shape = padded_shape
        self.features_out = features_out
        self.lambda_init = lambda_init
        self.stage_kwargs = stage_kwargs

        self.n_latents = len(config_stochastic)

        # input padding
        self.pad = None
        if padded_shape is not None:
            if not len(padded_shape) == len(input_shape[1:]):
                raise ValueError(f"'{padded_shape=}' and '{input_shape[1:]=}' must have same number of dimensions")
            padding = [[(t - o) // 2, (t - o) // 2] for t, o in zip(padded_shape, input_shape[1:])]
            self.pad = [u for pads in padding for u in pads]
            self.unpad = [-u for u in self.pad]
            input_shape = [input_shape[0], *padded_shape]

        # initialize the inference path
        stages_ = []
        block_args = {"activation": activation, "q_dropout": q_dropout, "p_dropout": p_dropout}

        stage_in_shape = {"x": input_shape}
        for i, (cfg_deterministic, cfg_stochastic) in enumerate(zip(config_deterministic, config_stochastic)):
            top = i == len(config_deterministic) - 1
            bottom = i == 0

            stage = Stage(
                in_shape=stage_in_shape,
                config_deterministic=cfg_deterministic,
                config_stochastic=cfg_stochastic,
                top=top,
                bottom=bottom,
                skip_stochastic=skip_stochastic,
                **block_args,
                **stage_kwargs,
            )

            stage_in_shape = stage.q_out_shape
            stages_ += [stage]

        self.stages = nn.ModuleList(stages_)

        # Likelihood
        likelihood = get_likelihood(likelihood_module)
        if features_out is None:
            features_out = input_shape[0]
        out_shape = (features_out, *input_shape[1:])
        input_shape = self.stages[0].p_out_shape["d"]
        kwargs = (
            {"weightnorm": config_deterministic[-1][-1]["weightnorm"]}
            if "weightnorm" in config_deterministic[-1][-1]
            else {}
        )
        self.likelihood = likelihood(input_shape, out_shape, activation=activation, **kwargs)

    @classmethod
    def get_argparser(cls, parents=[]):
        parser = argparse.ArgumentParser(description=cls.__name__, parents=parents, add_help=len(parents) == 0)
        parser.add_argument("--input_shape", default=None, type=tuple, help="")
        parser.add_argument("--likelihood_module", default=None, type=str, help="")
        parser.add_argument("--config_deterministic", type=json_file_or_json, default=None, help="")
        parser.add_argument("--config_stochastic", type=json_file_or_json, default=None, help="")
        parser.add_argument("--q_dropout", default=0.0, type=float, help="inference model dropout")
        parser.add_argument("--p_dropout", default=0.0, type=float, help="generative model dropout")
        parser.add_argument("--activation", default="ReLU", type=str, help="model activation function")
        parser.add_argument("--skip_stochastic", type=str2bool, default=True, help="skip connections between latents")
        parser.add_argument("--padded_shape", default=None, type=int, nargs="+", help="shape to which to pad the input")
        return parser

    def infer(
        self, x: torch.Tensor, use_mode: Union[bool, List[bool]] = False, **kwargs: Any
    ) -> Tuple[Union[Any, StochasticData]]:
        """
        Forward pass through the inference network and return the posterior of each layer order from the top to the bottom.
        :param x: input tensor
        :param use_mode: if True or list of True/False, we use the mode of the stochastic layer of those stages.
        :param kwargs: additional arguments passed to each stage
        :return: a list that contains the data for each stage
        """
        if self.pad is not None:
            x = nn.functional.pad(x, self.pad)

        if isinstance(use_mode, bool):
            use_mode = [use_mode] * len(self.stages)

        posteriors = []
        input_output = self.stages[0].IO(x=x)  # Create IO struct
        for i, stage in enumerate(self.stages):
            input_output, q_data = stage.infer(input_output, use_mode=use_mode[i], **kwargs)
            posteriors += [q_data]

        return tuple(posteriors)

    def generate(
        self,
        posteriors: Optional[List[Union[Any, StochasticData]]] = None,
        x: Optional[torch.Tensor] = None,
        use_mode: Union[bool, List[bool]] = False,
        decode_from_p: Union[bool, List[bool]] = False,
        forced_latent: Union[torch.Tensor, List[torch.Tensor]] = None,
        **stage_kwargs,
    ) -> Tuple[LikelihoodData, Tuple[StageData]]:
        """
        Forward pass through the generative model, compute KL and return reconstruction x_, KL and auxiliary data.
        If no posterior is provided, the prior is sampled.

        :param posteriors: a list containing the posterior for each stage
        :param use_mode: if True or list of True/False, we use the mode of the stochastic layer of those stages.
        :param decode_from_p: if true, use sample from p(z|-) for generation. Makes a difference only if posteriors
            are given as otherwise we already sample from p(z|-) as is standard for generation.
        :param stage_kwargs: additional arguments passed to each stage
        :return: tuple of LikelihoodData and list of StageData
        """
        if posteriors is None:
            posteriors = [None] * len(self.stages)

        if isinstance(use_mode, bool):
            use_mode = [use_mode] * len(self.stages)

        if isinstance(decode_from_p, bool):
            decode_from_p = [decode_from_p] * len(self.stages)

        if forced_latent is None:
            forced_latent = [forced_latent] * len(self.stages)

        stage_datas = []
        input_output = self.stages[-1].IO()  # For generation, no initial main input (sample prior or posterior)
        for i, stage in zip(reversed(range(len(self.stages))), self.stages[::-1]):
            input_output, stage_data = stage(
                io=input_output,
                posterior=posteriors[i],
                decode_from_p=decode_from_p[i],
                use_mode=use_mode[i],
                forced_latent=forced_latent[i],
                **stage_kwargs,
            )

            if isinstance(stage_data, list):
                stage_datas.extend(stage_data)
            else:
                stage_datas.append(stage_data)

        stage_datas = tuple(stage_datas[::-1])  # sort data: [z1, z2, ..., z_L]

        x_p = input_output.d

        if self.pad is not None:  # undo padding
            x_p = nn.functional.pad(x_p, self.unpad)

        _, likelihood_data = self.likelihood(x_p=x_p, x=x)

        return likelihood_data, stage_datas

    def forward(
        self,
        x: torch.Tensor,
        n_posterior_samples: int = 1,
        use_mode: Union[bool, List[bool]] = False,
        decode_from_p: Union[bool, List[bool]] = False,
        **stage_kwargs: Any,
    ) -> Tuple[LikelihoodData, List[StageData]]:
        """
        Forward pass through the inference model, the generative model and compute KL for each stage.
        x_ = p_\theta(x|z), z \sim q_\phi(z|x)
        kl_i = log q_\phi(z_i | h) - log p_\theta(z_i | h)
        :param x: input tensor
        :param n_posterior_samples: number of samples from the posterior distribution
        :param stage_kwargs: additional arguments passed to each stage
        :return: {'x_': reconstruction logits, 'kl': kl for each stage, **auxiliary}
        """
        x = x.repeat(n_posterior_samples, *((1,) * (x.ndim - 1)))  # Posterior samples

        if self.lambda_init is not None:
            x = self.lambda_init(x)

        posteriors = self.infer(x, use_mode=use_mode, **stage_kwargs)

        likelihood_data, stage_datas = self.generate(
            posteriors=posteriors,
            x=x,
            n_prior_samples=x.size(0),
            decode_from_p=decode_from_p,
            use_mode=use_mode,
            **stage_kwargs,
        )

        return likelihood_data, stage_datas

    @property
    def prior(self):
        return self.stages[-1].stochastic.prior

    @torch.no_grad()
    def sample_from_prior(
        self,
        n_prior_samples: int = 1,
        use_mode: Union[bool, List[bool]] = False,
        decode_from_p: Union[bool, List[bool]] = False,
        forced_latent: Union[torch.Tensor, List[torch.Tensor]] = None,
        **kwargs: Any,
    ) -> Tuple[LikelihoodData, List[StageData]]:
        """
        Sample the prior and pass through the generative model.
        x_ = p_\theta(x|z), z \sim p_\theta(z)
        :param n_prior_samples: number of samples (batch size)
        :param kwargs: additional arguments passed to each stage
        :return: {'x_': sample logits}
        """
        return self.generate(
            posteriors=None,
            use_mode=use_mode,
            decode_from_p=decode_from_p,
            forced_latent=forced_latent,
            n_prior_samples=n_prior_samples,
            **kwargs,
        )


class VAE(DeepVAE):
    def __init__(self, **kwargs):
        kwargs.pop("Stage", None)
        self.kwargs = kwargs
        super().__init__(Stage=VaeStage, **kwargs)


class LVAE(DeepVAE):
    def __init__(self, **kwargs):
        kwargs.pop("Stage", None)
        self.kwargs = kwargs
        super().__init__(Stage=LvaeStage, **kwargs)


class BIVA(DeepVAE):
    def __init__(self, **kwargs):
        kwargs.pop("Stage", None)
        self.kwargs = kwargs
        super().__init__(Stage=BivaStage, **kwargs)

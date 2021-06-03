"""
Stages are modules that hold 'DeterministicModule's and 'StochasticModule's 
and wire them together to define an inference network and a generative network.
"""
import logging

from copy import copy
from collections import namedtuple
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn

from torch import Tensor

from .stochastic import StochasticData, LossData, StochasticModuleConstructor
from .deterministic import DeterministicModuleConstructor, DeterministicModules, AsFeatureMap
from .activations import get_activation

from oodd.utils.shape import concatenate_shapes


LOGGER = logging.getLogger(name=__file__)


STAGE_METADATA_FIELDS = ["decode_from_p", "bu_inference"]
StageMetaData = namedtuple(
    typename="StageMetaData", field_names=STAGE_METADATA_FIELDS, defaults=[None] * len(STAGE_METADATA_FIELDS)
)

STAGE_DATA_FIELDS = ["q", "p", "loss", "metadata"]
StageData = namedtuple(typename="StageData", field_names=STAGE_DATA_FIELDS, defaults=[None] * len(STAGE_DATA_FIELDS))


class StageModule(nn.Module):
    IO = namedtuple("IO", ["x"])

    def __init__(
        self,
        in_shape: Dict[str, Tuple[int]],
        config_deterministic: List[Dict[str, Any]],
        config_stochastic: Dict[str, Any],
        top: bool = False,
        bottom: bool = False,
        activation: str = "ReLU",
        q_dropout: float = 0,
        p_dropout: float = 0,
        skip_stochastic: bool = True,
        **kwargs
    ):
        """
        Define a stage of a hierarchical model.
        In a VAE setting, a stage defines:
        - the latent variable z_i
        - the encoder q(z_i | h_{q<i})
        - the decoder p(z_{i-1} | z_i)
        """
        super().__init__()
        self._input_shape = in_shape
        self._config_deterministic = config_deterministic
        self._config_stochastic = config_stochastic
        self._top = top
        self._bottom = bottom
        self._skip_stochastic = skip_stochastic

    @property
    def in_shape(self) -> Dict[str, Tuple[int]]:
        """size of the input tensors for the inference path"""
        return self._input_shape

    @property
    def q_out_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        raise NotImplementedError

    @property
    def forward_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        raise NotImplementedError

    def infer(self, io: Dict[str, Tensor], **stochastic_kwargs) -> Tuple[IO, StochasticData]:
        """
        Perform a forward pass through the inference layers and sample the posterior.
        :param data: input data
        :param kwargs: additional parameters passed to the config_'StochasticModule'
        :return: (output data, variational data)
        """
        raise NotImplementedError

    def forward(
        self, io: Tuple[Tensor], posterior: Optional[StochasticData], decode_from_p: bool = False, **stochastic_kwargs
    ) -> Tuple[IO, StageData]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available
        :param data: data from the above stage forward pass
        :param posterior: dictionary representing the posterior from same stage inference pass
        :return: (dict('d' : d, 'aux : [aux]), dict('kl': [kl], **auxiliary) )
        """
        raise NotImplementedError

    def get_generative_parameters(self):
        raise NotImplementedError()

    def get_inference_parameters(self):
        raise NotImplementedError()


class VaeStage(StageModule):
    IO = namedtuple("IO", ["x", "aux", "d"], defaults=[None] * 3)

    def __init__(
        self,
        in_shape: Dict[str, Tuple[int]],
        config_deterministic: List[Dict[str, Any]],
        config_stochastic: Dict[str, Any],
        top: bool = False,
        bottom: bool = False,
        activation: str = "ReLU",
        q_dropout: float = 0,
        p_dropout: float = 0,
        skip_stochastic: bool = True,
        **kwargs
    ):
        """
        Conventional Variational Autoencoder stage with skip connections between latents in
        inference and generative networks [1].

                  q(z|x)           p(x|z)

                   +---+            +---+
                   |z_3|            |z_3|
                   +---+            +---+
                     ^                |
                     |                |
                +--->|           |    v
                |  +---+         |  +---+
             aux|  |z_2|      aux|  |z_2|
                |  +---+         |  +---+
                |    ^           +--->|
                     |                |
                +--->|           |    v
                |  +---+         |  +---+
             aux|  |z_1|      aux|  |z_1|
                |  +---+         |  +---+
                |    ^           +--->|
                     |                |
                     |                v
                   +---+            +---+
                   | x |            | x |
                   +---+            +---+

        Defines a Variational Autoencoder stage containing:
        - a sequence of 'DeterministicModule's for the inference model
        - a sequence of 'DeterministicModule's for the generative model
        - a 'StochasticModule'

        :param in_shape: dictionary describing the input tensors of shapes (B, H, *D)
        :param config_deterministic: list of tuple describing a 'DeterministicModule' (filters, kernel_size, stride)
        :param config_stochastic: integer or tuple describing the 'StochasticModule': units or (units, kernel_size, discrete, K)
        :param top: whether this is the top stage
        :param bottom: whether this is the bottom stage
        :param activation: the activation function to use
        :param q_dropout: inference model dropout value
        :param p_dropout: generative model dropout value
        :param skip_stochastic: use skip connections over latent variables if True
        :param kwargs: others arguments passed to the block constructors (both deterministic and stochastic)

        [1]: https://arxiv.org/abs/1312.6114
        """
        super().__init__(
            in_shape,
            config_deterministic,
            config_stochastic,
            top=top,
            bottom=bottom,
            q_dropout=q_dropout,
            p_dropout=p_dropout,
            skip_stochastic=skip_stochastic,
        )

        x_shape = in_shape.get("x")
        aux_shape = in_shape.get("aux", None)

        activation = get_activation(activation)

        # mute skip connections
        if not skip_stochastic:
            aux_shape = None

        # define inference 'DeterministicModule's
        in_residual = not bottom
        q_skips = [aux_shape for _ in config_deterministic] if aux_shape is not None else None
        self.q_deterministic = DeterministicModules(
            x_shape,
            config_deterministic,
            aux_shape=q_skips,
            transposed=False,
            in_residual=in_residual,
            dropout=q_dropout,
            activation=activation,
            **kwargs
        )

        # shape of the deterministic output
        x_shape = self.q_deterministic.out_shape

        # define the config_'StochasticModule'
        self.stochastic = StochasticModuleConstructor(
            config_stochastic, in_shape=x_shape, top=top, activation=activation, **kwargs
        )

        self.q_projection = AsFeatureMap(self.stochastic.out_shape, self.stochastic.in_shape)
        self._q_out_shape = {"x": self.q_projection.out_shape, "aux": x_shape}

        # project z sample
        self.p_projection = AsFeatureMap(self.stochastic.out_shape, self.stochastic.in_shape)

        # define the generative 'DeterministicModule's with the skip connections
        # This should work as long as the number of channels is constant throughout the network.
        # This is required since we assume the skip connections to be of the same shape as `x_shape`: this does not work
        # with every configuration of the generative model. Making the arhitecture more general requires having
        # a top-down __init__() method such as to take the shapes of the above generative block skip connections as input.
        # Specifically, we need to know 'self.p_deterministic.hidden_shapes/out_shape' but for the Stage above this one.
        p_skips = None if (top or not skip_stochastic) else [x_shape] * len(config_deterministic)
        self.p_deterministic = DeterministicModules(
            self.p_projection.out_shape,
            config_deterministic[::-1],
            aux_shape=p_skips,
            transposed=True,
            in_residual=False,
            dropout=p_dropout,
            activation=activation,
            **kwargs
        )
        self._p_out_shape = {"d": self.p_deterministic.out_shape, "aux": self.p_deterministic.hidden_shapes}

    @property
    def q_out_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._q_out_shape

    @property
    def p_out_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._p_out_shape

    @property
    def forward_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        raise NotImplementedError

    def infer(self, io: IO, **stochastic_kwargs) -> Tuple[IO, StochasticData]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param io: input data
        :param stochastic_kwargs: additional parameters passed to the 'StochasticModule'
        :return: (output data, variational data)
        """
        x = io.x
        aux = [io.aux] * len(self.q_deterministic) if self._skip_stochastic else None

        h, _ = self.q_deterministic(x, aux)

        z, q_data = self.stochastic.infer(h, **stochastic_kwargs)
        z = self.q_projection(z)

        io = VaeStage.IO(x=z, aux=h)
        return io, q_data

    def forward(
        self, io: IO, posterior: Optional[StochasticData], decode_from_p: bool = False, **stochastic_kwargs
    ) -> Tuple[IO, StageData]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param io: output from the above stage forward pass used as input here
        :param posterior: dictionary representing the posterior from same stage inference pass
        :param decode_from_p: pass the generative sample z~p(z|-) through this stage instead of z~q(z|-)
        :param stochastic_kwargs: additional parameters passed to the 'StochasticModule'
        :return: (dict('d' : d, 'aux : [aux]), dict('kl': [kl], **auxiliary) )
        """
        d = io.d
        aux = io.aux if self._skip_stochastic else None

        # sample p(z | d)
        z_p, p_data = self.stochastic.generate(d, sample=posterior is None or decode_from_p, **stochastic_kwargs)

        # sample from p if q is not available or forced, otherwise from q and compute KL
        if posterior is None or decode_from_p:
            z = z_p
            loss_data = LossData()
        else:
            z = posterior.z
            loss_data = self.stochastic.loss(posterior, p_data)

        # project z
        z = self.p_projection(z)

        # pass through deterministic blocks
        d, aux = self.p_deterministic(z, aux=aux)

        stage_data = StageData(
            q=posterior, p=p_data, loss=loss_data, metadata=StageMetaData(decode_from_p=decode_from_p)
        )
        io = VaeStage.IO(d=d, aux=aux)
        return io, stage_data

    def get_generative_parameters(self):
        yield from self.p_deterministic.parameters()
        yield from self.p_projection.parameters()
        yield from self.stochastic.get_generative_parameters()

    def get_inference_parameters(self):
        yield from self.q_deterministic.parameters()
        yield from self.q_projection.parameters()
        yield from self.stochastic.get_inference_parameters()


class LvaeStage(VaeStage):
    IO = namedtuple("IO", ["x", "aux", "d", "h"], defaults=[None] * 4)
    DeterministicData = namedtuple("DeterministicData", ["h"])

    def __init__(
        self,
        in_shape: Dict[str, Tuple[int]],
        config_deterministic: List[Dict[str, Any]],
        config_stochastic: Dict[str, Any],
        top: bool = False,
        bottom: bool = False,
        activation: str = "ReLU",
        q_dropout: float = 0,
        p_dropout: float = 0,
        skip_stochastic: bool = True,
        **kwargs
    ):
        """
        LVAE: https://arxiv.org/abs/1602.02282

        Define a Ladder Variational Autoencoder stage containing:
        - a sequence of 'DeterministicModule's for the inference model
        - a sequence of 'DeterministicModule's for the generative model
        - a config_'StochasticModule'

                q(z|x)            p(x|z)

            +---+     +---+        +---+
            |d_3|---->|z_3|        |z_3|
            +---+     +---+        +---+
              ^         |            |
              |         v            v
            +---+     +---+        +---+
            |d_2|---->|z_2|        |z_2|
            +---+     +---+        +---+
              ^         |            |
              |         v            v
            +---+     +---+        +---+
            |d_1|---->|z_1|        |z_1|
            +---+     +---+        +---+
              ^                      |
              |                      v
            +---+                  +---+
            | x |                  | x |
            +---+                  +---+

        :param in_shape: dictionary describing the input tensors of shapes (B, H, *D)
        :param convolution: list of tuple describing a 'DeterministicModule' (filters, kernel_size, stride)
        :param config_stochastic: integer or tuple describing the config_'StochasticModule': units or (units, kernel_size, discrete, K)
        :param top: is top layer
        :param bottom: is bottom layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param kwargs: others arguments passed to the block constructors (both config_deterministic and config_stochastic)
        """
        super().__init__(
            in_shape,
            config_deterministic=config_deterministic,
            config_stochastic=config_stochastic,
            top=top,
            bottom=bottom,
            activation=activation,
            q_dropout=q_dropout,
            p_dropout=p_dropout,
            skip_stochastic=skip_stochastic,
            **kwargs
        )

        # Modify the VaeStage:
        self.q_projection = None
        top_shape = self._q_out_shape.get("aux")  # get the tensor shape of the output of the deterministic path
        self._q_out_shape["x"] = top_shape  # modify the output of the inference path to be only deterministic

        # Build merge operation identical to last layer of deterministic transform but without stride
        activation = get_activation(activation)
        top_down_shape = top_shape if not top else None
        merge_config = config_deterministic[-1].copy()
        if "stride" in merge_config:
            merge_config["stride"] = 1  # Force stride 1 on the merge operation

        self.merge = DeterministicModuleConstructor(
            config=merge_config,
            in_shape=top_shape,
            aux_shape=top_down_shape,
            transposed=False,
            residual=True,
            activation=activation,
            dropout=p_dropout,
            **kwargs
        )

    def infer(self, io: IO, **stochastic_kwargs) -> Tuple[IO, DeterministicData]:
        """
        Perform a forward pass through the deterministic inference layers and return deterministic path output.

        :param io: input io
        :param stochastic_kwargs: additional parameters passed to the config_'StochasticModule'
        :return: (output io, deterministic io)
        """
        x = io.x
        aux = io.aux
        if not self._skip_stochastic:
            aux = None

        aux = [aux] * len(self.q_deterministic) if aux is not None else None
        x, _ = self.q_deterministic(x, aux)

        return LvaeStage.IO(x=x, aux=x), LvaeStage.DeterministicData(h=x)

    def forward(
        self, io: IO, posterior: Optional[DeterministicData], decode_from_p: bool = False, **stochastic_kwargs
    ) -> Tuple[IO, StageData]:
        """
        Perform a forward pass through the generative model and compute KL if posterior io is available

        :param io: io from the above stage forward pass
        :param posterior: dictionary representing the posterior
        :return: (dict('d' : d, 'aux : [aux]), dict('kl': [kl], **auxiliary) )
        """
        d = io.d  # Top most is always None

        # sample p(z | d)
        z_p, p_data = self.stochastic(
            d, inference=False, sample=posterior is None or decode_from_p, **stochastic_kwargs
        )

        if posterior is None or decode_from_p:
            # sample p(z) without computing KL(q | p)
            q_data = None
            loss_data = LossData()
            z = z_p
        else:
            # sample q(z | h) and compute KL(q | p)
            # compute the top-down logits of q(z_i | x, z_{>i})
            h = posterior.h
            h = self.merge(h, aux=d)

            # z ~ q(z | h_bu, d_td)
            z_q, q_data = self.stochastic(h, inference=True, **stochastic_kwargs)
            loss_data = self.stochastic.loss(q_data, p_data)
            z = z_q

        # project z
        z = self.p_projection(z)

        # pass through deterministic
        aux = io.aux
        if not self._skip_stochastic:
            aux = None

        d, skips = self.p_deterministic(z, aux)

        stage_data = StageData(q=q_data, p=p_data, loss=loss_data, metadata=StageMetaData(decode_from_p=decode_from_p))
        return LvaeStage.IO(d=d, aux=skips), stage_data


class BivaIntermediateStage(StageModule):
    IO = namedtuple("IO", ["x", "x_td", "x_bu", "aux", "d", "h"], defaults=[None] * 6)

    def __init__(
        self,
        in_shape: Dict[str, Tuple[int]],
        config_deterministic: List[Dict[str, Any]],
        config_stochastic: Dict[str, Any],
        top: bool = False,
        bottom: bool = False,
        activation: str = "ReLU",
        q_dropout: float = 0,
        p_dropout: float = 0,
        skip_stochastic: bool = True,
        conditional_bu: bool = False,
        **kwargs
    ):
        """
        BIVA: https://arxiv.org/abs/1902.02102

        Define a Bidirectional Variational Autoencoder stage containing:
        - a sequence of 'DeterministicModule's for the bottom-up inference model (BU)
        - a sequence of 'DeterministicModule's for the top-down inference model (TD)
        - a sequence of 'DeterministicModule's for the generative model
        - two config_'StochasticModule's (BU and TD)

        :param in_shape: dictionary describing the input tensor shape (B, H, *D)
        :param convolution: list of tuple describing a 'DeterministicModule' (filters, kernel_size, stride)
        :param config_stochastic: dictionary describing the config_'StochasticModule': units or (units, kernel_size, discrete, K)
        :param bottom: is bottom layer
        :param top: is top layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param skip_stochastic: do not use skip connections
        :param conditional_bu: condition BU prior on p(z_TD)
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param kwargs: others arguments passed to the block constructors (both config_deterministic and config_stochastic)
        """
        super().__init__(
            in_shape,
            config_deterministic,
            config_stochastic,
            top=top,
            bottom=bottom,
            q_dropout=q_dropout,
            p_dropout=p_dropout,
            skip_stochastic=skip_stochastic,
        )

        self._conditional_bu = conditional_bu
        activation = get_activation(activation)

        if "x" in in_shape.keys():
            bu_shp = td_shp = in_shape.get("x")
            aux_shape = None
        else:
            bu_shp = in_shape.get("x_bu")
            td_shp = in_shape.get("x_td")
            aux_shape = in_shape.get("aux")

        if isinstance(config_stochastic.get("block"), tuple):
            bu_block, td_block = config_stochastic.get("block")
            bu_stochastic = copy(config_stochastic)
            td_stochastic = copy(config_stochastic)
            bu_stochastic["block"] = bu_block
            td_stochastic["block"] = td_block
        else:
            bu_stochastic = td_stochastic = config_stochastic

        # mute skip connections
        if skip_stochastic:
            aux_shape = None

        # define inference 'DeterministicModule's
        in_residual = not bottom
        q_bu_aux = [aux_shape for _ in config_deterministic] if aux_shape is not None else None
        self.q_bu_convs = DeterministicModules(
            bu_shp,
            config_deterministic,
            aux_shape=q_bu_aux,
            transposed=False,
            in_residual=in_residual,
            dropout=q_dropout,
            activation=activation,
            **kwargs
        )

        q_td_aux = [self.q_bu_convs.out_shape for _ in config_deterministic]
        self.q_td_convs = DeterministicModules(
            td_shp,
            config_deterministic,
            aux_shape=q_td_aux,
            transposed=False,
            in_residual=in_residual,
            dropout=q_dropout,
            activation=activation,
            **kwargs
        )

        # shape of the output of the inference path and input tensor from the generative path
        top_tensor_shp = self.q_td_convs.out_shape
        aux_shape = concatenate_shapes([top_tensor_shp, top_tensor_shp], 0)

        # define the BU StochasticModule
        bu_top = False if conditional_bu else top
        self.bu_stochastic = StochasticModuleConstructor(bu_stochastic, top_tensor_shp, top=bu_top, **kwargs)
        self.bu_proj = AsFeatureMap(self.bu_stochastic.out_shape, self.bu_stochastic.in_shape, **kwargs)

        # define the TD StochasticModule
        self.td_stochastic = StochasticModuleConstructor(td_stochastic, top_tensor_shp, top=top, **kwargs)

        self._q_out_shape = {"x_bu": self.bu_proj.out_shape, "x_td": top_tensor_shp, "aux": aux_shape}

        ### GENERATIVE MODEL
        # TD merge layer
        h_shape = self._q_out_shape.get("x_td", None) if not self._top else None
        merge_config = config_deterministic[-1].copy()
        if "stride" in merge_config:
            merge_config["stride"] = 1  # Force stride 1 on the merge operation

        self.merge = DeterministicModuleConstructor(
            config=merge_config,
            in_shape=h_shape,
            aux_shape=h_shape,
            transposed=False,
            residual=True,
            activation=activation,
            dropout=p_dropout,
            **kwargs
        )

        # alternative: define the condition p(z_bu | z_td, ...)
        if conditional_bu:
            self.bu_condition = DeterministicModuleConstructor(
                self.bu_stochastic.out_shape,
                merge_config,
                aux_shape=h_shape,
                transposed=False,
                in_residual=False,
                dropout=p_dropout,
                **kwargs
            )
        else:
            self.bu_condition = None

        # merge latent variables
        z_shape = concatenate_shapes([self.bu_stochastic.out_shape, self.td_stochastic.out_shape], 0)
        self.z_proj = AsFeatureMap(z_shape, self.bu_stochastic.in_shape)

        # define the generative 'DeterministicModule's with the skip connections
        # here we assume the skip connections to be of the same shape as `top_tensor_shape` : this does not work with
        # with every configuration of the generative model. Making the arhitecture more general requires to have
        # a top-down __init__() method such as to take the shapes of the above generative block skip connections as input.
        p_skips = None if (top or not skip_stochastic) else [top_tensor_shp] * len(config_deterministic)
        self.p_deterministic = DeterministicModules(
            self.z_proj.out_shape,
            config_deterministic,
            aux_shape=p_skips,
            transposed=True,
            activation=activation,
            in_residual=False,
            dropout=p_dropout,
            **kwargs
        )

        self._p_out_shape = {"d": self.p_deterministic.out_shape, "aux": self.p_deterministic.hidden_shapes}

    @property
    def q_out_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._q_out_shape

    @property
    def p_out_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._p_out_shape

    def infer(self, io: IO, **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the config_'StochasticModule'
        :return: (output data, variational data)
        """
        if io.x is not None:
            x = io.x
            x_bu, x_td = x, x
        else:
            x_bu = io.x_bu
            x_td = io.x_td

        aux = io.aux
        if not self._skip_stochastic:
            aux = None

        # BU path
        bu_aux = [aux for _ in range(len(self.q_bu_convs))] if aux is not None else None
        x_bu, _ = self.q_bu_convs(x_bu, aux=bu_aux)
        # z_bu ~ q(x)
        z_bu, bu_q_data = self.bu_stochastic(x_bu, inference=True, **kwargs)
        z_bu_proj = self.bu_proj(z_bu)

        # TD path
        td_aux = [x_bu for _ in range(len(self.q_td_convs))]
        x_td, _ = self.q_td_convs(x_td, aux=td_aux)
        td_q_data = {"z": z_bu, "h": x_td}  # note h = d_q(x)

        # skip connection
        aux = torch.cat([x_bu, x_td], 1)

        return BivaIntermediateStage.IO(x_bu=z_bu_proj, x_td=x_td, aux=aux), {
            "z_bu": z_bu,
            "bu": bu_q_data,
            "td": td_q_data,
        }

    def forward(
        self, io: IO, posterior: Optional[dict], decode_from_p: bool = False, **stochastic_kwargs
    ) -> Tuple[IO, List[StageData]]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param d: previous hidden state
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl], **auxiliary))
        """
        d = io.d

        # TODO Make `decode_from_p` act as in the VAEStage above (compute no KL loss)
        if decode_from_p:
            LOGGER.warning(
                "Using decode_from_p=%s in BivaIntermediateStage but this is not correctly implemented!", decode_from_p
            )

        if posterior is None or decode_from_p:
            # sample priors
            # top-down
            z_td_p, td_p_data = self.td_stochastic(d, inference=False, sample=True, **stochastic_kwargs)  # prior

            # conditional BU prior
            if self.bu_condition is not None:
                d_ = self.bu_condition(z_td_p, aux=d)
            else:
                d_ = d

            # bottom-up
            z_bu_p, bu_p_data = self.bu_stochastic(d_, inference=False, sample=True, **stochastic_kwargs)  # prior

            # merge samples
            z = torch.cat([z_td_p, z_bu_p], 1)

            # null loss and approximate posteriors
            bu_loss_data, td_loss_data = LossData(), LossData()
            bu_q_data, td_q_data = None, None
        else:
            # sample posterior and compute KL using prior
            bu_q_data = posterior.get("bu")
            td_q_data = posterior.get("td")
            z_bu_q = posterior.get("z_bu")

            # top-down: compute the posterior using the bottom-up hidden state and top-down hidden state
            # p(z_td | d_top)
            _, td_p_data = self.td_stochastic(d, inference=False, sample=False, **stochastic_kwargs)

            # merge d_top with h = d_q(x)
            h = td_q_data.get("h")
            h = self.merge(h, aux=d)

            # z_td ~ q(z_td | h_bu_td)
            z_td_q, td_q_data = self.td_stochastic(h, inference=True, sample=True, **stochastic_kwargs)

            # compute log q_bu(z_i | x) - log p_bu(z_i) (+ additional data)
            td_loss_data = self.td_stochastic.loss(td_q_data, td_p_data)

            # conditional BU prior
            if self.bu_condition is not None:
                d_ = self.bu_condition(z_td_q, aux=d)
            else:
                d_ = d

            # bottom-up: retrieve data from the inference path
            # z_bu ~ p(d_top)
            _, bu_p_data = self.bu_stochastic(d_, inference=False, sample=False, **stochastic_kwargs)

            # compute log q_td(z_i | x, z_{>i}) - log p_td(z_i) (+ additional data)
            bu_loss_data = self.bu_stochastic.loss(bu_q_data, bu_p_data)

            # merge samples
            z = torch.cat([z_td_q, z_bu_q], 1)

        # projection
        z = self.z_proj(z)

        # pass through config_deterministic
        aux = io.aux
        if not self._skip_stochastic:
            aux = None

        d, skips = self.p_deterministic(z, aux=aux)

        # gather data
        stage_data_bu = StageData(
            q=bu_q_data,
            p=bu_p_data,
            loss=bu_loss_data,
            metadata=StageMetaData(decode_from_p=decode_from_p, bu_inference=True),
        )
        stage_data_td = StageData(
            q=td_q_data,
            p=td_p_data,
            loss=td_loss_data,
            metadata=StageMetaData(decode_from_p=decode_from_p, bu_inference=False),
        )
        stage_out = [stage_data_td, stage_data_bu]
        io = BivaIntermediateStage.IO(d=d, aux=skips)
        return io, stage_out


class BivaTopStage(StageModule):
    IO = namedtuple("IO", ["x", "x_td", "x_bu", "aux", "d", "h"], defaults=[None] * 6)

    def __init__(
        self,
        in_shape: Dict[str, Tuple[int]],
        config_deterministic: List[Dict[str, Any]],
        config_stochastic: Dict[str, Any],
        bottom: bool = False,
        activation: str = "ReLU",
        q_dropout: float = 0,
        p_dropout: float = 0,
        skip_stochastic: bool = True,
        **kwargs
    ):
        """
        BIVA: https://arxiv.org/abs/1902.02102

        Define a Bidirectional Variational Autoencoder top stage containing:
        - a sequence of 'DeterministicModule's for the bottom-up inference model (BU)
        - a sequence of 'DeterministicModule's for the top-down inference model (TD)
        - a 'DeterministicModule' to merge BU and TD
        - a sequence of 'DeterministicModule's for the generative model
        - a config_'StochasticModule' (z_L)

        :param in_shape: dictionary describing the input tensor shape (B, H, *D)
        :param convolution: list of tuple describing a 'DeterministicModule' (filters, kernel_size, stride)
        :param config_stochastic: dictionary describing the config_'StochasticModule': units or (units, kernel_size, discrete, K)
        :param bottom: is bottom layer
        :param top: is top layer
        :param q_dropout: inference dropout value
        :param p_dropout: generative dropout value
        :param skip_stochastic: do not use skip connections
        :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
        :param kwargs: others arguments passed to the block constructors (both config_deterministic and config_stochastic)
        """
        super().__init__(
            in_shape,
            config_deterministic,
            config_stochastic,
            top=True,
            bottom=bottom,
            q_dropout=q_dropout,
            p_dropout=p_dropout,
            skip_stochastic=skip_stochastic,
        )

        kwargs.pop("top")
        top = True

        activation = get_activation(activation)

        if "x" in in_shape.keys():
            bu_shp = td_shp = in_shape.get("x")
            aux_shape = None
        else:
            bu_shp = in_shape.get("x_bu")
            td_shp = in_shape.get("x_td")
            aux_shape = in_shape.get("aux")

        # mute skip connections
        if not skip_stochastic:
            aux_shape = None

        # define inference BU and TD paths
        in_residual = not bottom
        q_bu_aux = [aux_shape for _ in config_deterministic] if aux_shape is not None else None
        self.q_bu_convs = DeterministicModules(
            bu_shp,
            config_deterministic,
            aux_shape=q_bu_aux,
            transposed=False,
            activation=activation,
            in_residual=in_residual,
            dropout=q_dropout,
            **kwargs
        )

        q_td_aux = [self.q_bu_convs.out_shape for _ in config_deterministic]
        self.q_td_convs = DeterministicModules(
            td_shp,
            config_deterministic,
            aux_shape=q_td_aux,
            transposed=False,
            activation=activation,
            in_residual=in_residual,
            dropout=q_dropout,
            **kwargs
        )

        # merge BU and TD paths
        merge_config = config_deterministic[-1].copy()
        if "stride" in merge_config:
            merge_config["stride"] = 1  # Force stride 1 on the merge operation
        top_in_shape = concatenate_shapes([self.q_bu_convs.out_shape, self.q_td_convs.out_shape], 0)

        self.q_top = DeterministicModuleConstructor(
            config=merge_config,
            in_shape=top_in_shape,
            transposed=False,
            residual=True,
            activation=activation,
            dropout=q_dropout,
            **kwargs
        )

        top_tensor_shp = self.q_top.out_shape

        # config_'StochasticModule'
        self.stochastic = StochasticModuleConstructor(config_stochastic, top_tensor_shp, top=top, **kwargs)

        self._q_out_shape = {}  # no output shape (top layer)

        ### GENERATIVE MODEL

        # map sample back to a feature map
        self.z_proj = AsFeatureMap(self.stochastic.out_shape, self.stochastic.in_shape)

        # define the generative 'DeterministicModule's with the skip connections
        p_skips = None
        self.p_deterministic = DeterministicModules(
            self.z_proj.out_shape,
            config_deterministic,
            aux_shape=p_skips,
            transposed=True,
            activation=activation,
            in_residual=False,
            dropout=p_dropout,
            **kwargs
        )

        self._p_out_shape = {"d": self.p_deterministic.out_shape, "aux": self.p_deterministic.hidden_shapes}

    def infer(self, io: IO, **kwargs) -> Tuple[Dict[str, Any], StochasticData]:
        """
        Perform a forward pass through the inference layers and sample the posterior.

        :param data: input data
        :param kwargs: additional parameters passed to the config_'StochasticModule'
        :return: (output data, variational data)
        """

        if io.x is not None:
            x = io.x
            x_bu, x_td = x, x
        else:
            x_bu = io.x_bu
            x_td = io.x_td

        aux = io.aux
        if not self._skip_stochastic:
            aux = None

        # BU path
        bu_aux = [aux for _ in range(len(self.q_bu_convs))] if aux is not None else None
        x_bu, _ = self.q_bu_convs(x_bu, aux=bu_aux)

        # TD path
        td_aux = [x_bu for _ in range(len(self.q_td_convs))]
        x_td, _ = self.q_td_convs(x_td, aux=td_aux)

        # merge BU and TD
        x = torch.cat([x_bu, x_td], 1)
        x = self.q_top(x)

        # sample top layer
        z, q_data = self.stochastic(x, inference=True, **kwargs)

        return {}, q_data

    def forward(self, io: IO, posterior: Optional[dict], decode_from_p: bool = False, **kwargs) -> Tuple[IO, LossData]:
        """
        Perform a forward pass through the generative model and compute KL if posterior data is available

        :param data: data from the above stage forward pass
        :param posterior: dictionary representing the posterior
        :return: (hidden state, dict('kl': [kl], **auxiliary) )
        """
        d = io.d

        if posterior is None or decode_from_p:
            loss_data = LossData()
            z, p_data = self.stochastic(d, inference=False, sample=True, **kwargs)
        else:
            # get p(z | d)
            _, p_data = self.stochastic(d, inference=False, sample=False, **kwargs)

            # compute KL(q | p)
            loss_data = self.stochastic.loss(posterior, p_data)
            z = posterior.z

        # project z
        z = self.z_proj(z)

        # pass through config_deterministic
        aux = io.aux
        if not self._skip_stochastic:
            aux = None

        d, skips = self.p_deterministic(z, aux=aux)

        io = BivaTopStage.IO(d=d, aux=skips)
        stage_data = StageData(q=posterior, p=p_data, loss=loss_data)
        return io, stage_data

    @property
    def q_out_shape(self) -> Dict[str, Tuple[int]]:
        """size of the output tensors for the inference path"""
        return self._q_out_shape

    @property
    def p_out_shape(self) -> Tuple[int]:
        """size of the output tensor for the generative path"""
        return self._p_out_shape


class BivaTopStage_simpler(VaeStage):
    """
    This is the BivaTopStage without the additional BU-TD merge layer.
    """

    def __init__(self, in_shape: Dict[str, Tuple[int]], *args, **kwargs):
        bu_shp = in_shape.get("x_bu")
        td_shp = in_shape.get("x_td")

        x_shape = concatenate_shapes([bu_shp, td_shp], 0)
        concat_shape = {"x": x_shape}

        super().__init__(concat_shape, *args, **kwargs)

    def infer(self, data: Dict[str, Tensor], **kwargs) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        x_bu = data.pop("x_bu")
        x_td = data.pop("x_td")
        data["x"] = torch.cat([x_bu, x_td], 1)

        return super().infer(data, **kwargs)


def BivaStage(
    in_shape: Dict[str, Tuple[int]],
    config_deterministic: List[List[Dict[str, Any]]],
    config_stochastic: Dict[str, Any],
    top: bool = False,
    **kwargs
):
    """
    BIVA: https://arxiv.org/abs/1902.02102

    Define a Bidirectional Variational Autoencoder stage containing:
    - a sequence of 'DeterministicModule's for the bottom-up inference model (BU)
    - a sequence of 'DeterministicModule's for the top-down inference model (TD)
    - a sequence of 'DeterministicModule's for the generative model
    - two config_'StochasticModule's (BU and TD)

    This is not an op-for-op implementation of the original Tensorflow version.

    :param in_shape: dictionary describing the input tensor shape (B, H, *D)
    :param convolution: list of tuple describing a 'DeterministicModule' (filters, kernel_size, stride)
    :param config_stochastic: dictionary describing the config_'StochasticModule': units or (units, kernel_size, discrete, K)
    :param top: is top layer
    :param bottom: is bottom layer
    :param q_dropout: inference dropout value
    :param p_dropout: generative dropout value
    :param conditional_bu: condition BU prior on p(z_TD)
    :param aux_shape: auxiliary input tensor shape as a tuple of integers (B, H, *D)
    :param kwargs: others arguments passed to the block constructors (both config_deterministic and config_stochastic)
    """

    if top:
        return BivaTopStage(in_shape, config_deterministic, config_stochastic, top=top, **kwargs)
    else:
        return BivaIntermediateStage(in_shape, config_deterministic, config_stochastic, top=top, **kwargs)

import torch
import torch.nn as nn


class Swish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = beta

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class PSwish(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.beta = nn.Parameter(torch.Tensor([beta]))

    def forward(self, x):
        return x * torch.sigmoid(self.beta * x)


class InverseSoftplus(nn.Module):
    def __init__(self, beta: float = 1, threshold: float = 20):
        """The inverse transform of the SoftPlus activation"""
        super().__init__()
        self.register_buffer("beta", torch.FloatTensor([beta]))
        self.register_buffer("threshold", torch.FloatTensor([threshold]))

    def forward(self, x):
        return torch.where(x > self.threshold, x, (self.beta * x).expm1().log() / self.beta)


ACTIVATIONS = dict(
    Identity=nn.Identity,
    ReLU=nn.ReLU,
    LeakyReLU=nn.LeakyReLU,
    SELU=nn.SELU,
    ELU=nn.ELU,
    SoftPlus=nn.Softplus,
    InverseSoftPlus=InverseSoftplus,
    Tanh=nn.Tanh,
    Swish=Swish,
    PSwish=PSwish,
)


def get_activation(name):
    if not name in ACTIVATIONS:
        raise ValueError(f"Activation `{name}` not recognized")
    return ACTIVATIONS[name]

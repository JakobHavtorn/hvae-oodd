import pytest

import torch

from oodd.layers.stochastic import GaussianStochasticModule, GaussianDense, GaussianConv2d


def test_gaussian_stochastic_module_prior_is_standard_normal():
    gauss = GaussianStochasticModule(
        in_shape=128,
        latent_features=64,
        top=True
    )
    
    assert not hasattr(gauss, 'prior_logits')

    gauss.define_prior(64)

    assert hasattr(gauss, 'prior_logits')
    
    assert (gauss.prior.mean == torch.zeros_like(gauss.prior.mean)).all()
    assert (gauss.prior.stddev == torch.ones_like(gauss.prior.stddev)).all()

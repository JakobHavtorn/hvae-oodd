from typing import Union, List

import torch
import torch.nn as nn

from oodd.utils import reduce_batch, reduce_samples, reduce_to_batch, log_sum_exp
from oodd.variational import FreeNats


class ELBO(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        likelihood,
        kl_divergences,
        samples=1,
        beta=1,
        free_nats: Union[float, List[float]] = None,
        analytical_kl=False,
        batch_reduction=None,
        sample_reduction=log_sum_exp,
    ):
        """Compute the variational lower bound on the data log-likelihood (ELBO) of the VAE.

        Args:
            likelihood (torch.Tensor): Samplewise likelihood (samples * batch,)
            kl_divergences (list(torch.Tensor)): Samplewise KL divergence (samples * batch,) per stochastic layer.
            samples (int, optional): Number of posterior samples to sample. Defaults to 1.
            beta (int, optional): Value of ß for the ß-VAE or deterministic warmup parameter. Defaults to 1.
            free_nats (float, optional): Free nats (discounted nats) in each of the KL terms
            sample_reduction (callable, optional): Method for reducing posterior samples. If mean, we MC sample, if log_sum_exp, we importance sample. Defaults to log_sum_exp.
            analytical_kl (bool, optional): If `True`, use the analytical calculation of the KL divergence. Defaults to False.

        Returns:
            tuple: ELBO, likelihood and KL divergence. log p(x), log p(x|z) and KL(q(z|x), p(z))

        TODO Add IW and MC samples distinction
        """
        if analytical_kl and sample_reduction == log_sum_exp:
            raise ValueError("KL is not analytical under importance sampling (`sample_reduction == log_sum_exp`)")

        # Remove any KL divergences that are None.
        kl_divergences = [kl for kl in kl_divergences if kl is not None]

        if not isinstance(free_nats, list):
            free_nats = [free_nats] * len(kl_divergences)

        # Apply FreeNats to get KL divergence loss
        kl_losses = [FreeNats(fn, 1)(kl) for fn, kl in zip(free_nats, kl_divergences)]
        kl_loss = sum([reduce_to_batch(kl) for kl in kl_losses])
        kl_divergence = sum([reduce_to_batch(kl) for kl in kl_divergences])

        elbo = likelihood - kl_divergence  # Variational Lower Bound (ELBO)
        loss = -likelihood + beta * kl_loss  # Negative VLB with optional free nats and warmup of KL term

        loss, elbo, likelihood, kl_loss, kl_divergence = self.reduce_samples(
            loss, elbo, likelihood, kl_loss, kl_divergence, samples=samples, reduction=sample_reduction
        )
        loss, elbo, likelihood, kl_loss, kl_divergence = self.reduce_batches(
            loss, elbo, likelihood, kl_loss, kl_divergence, reduction=batch_reduction
        )

        return loss, elbo, likelihood, kl_divergence

    def reduce_samples(self, loss, elbo, *other, samples=1, reduction=log_sum_exp):
        """Reduce samples either as Monte Carlo Samples (`torch.mean`) or Importance Weighted Samples (`log_sum_exp`).

        The given tensors must have shape (samples, batch, *dimensions).
        The returned tensors have shape (batch, *dimensions).
        """
        if reduction is not None and samples > 1:
            loss = reduce_samples(loss, batch_dim=0, sample_dim=0, n_samples=samples, reduction=reduction)
            elbo = reduce_samples(elbo, batch_dim=0, sample_dim=0, n_samples=samples, reduction=reduction)

            other = list(other)
            for i in range(len(other)):
                other[i] = reduce_samples(other[i], batch_dim=0, sample_dim=0, n_samples=samples, reduction=torch.sum)

        return loss, elbo, *other

    def reduce_batches(self, loss, elbo, *other, reduction=torch.sum):
        """Reduce batch examples, typically via summation.

        The given tensors must have shape (samples, *dimensions).
        The returned tensors have shape (*dimensions).
        """
        if reduction is not None:
            loss = reduce_batch(loss, batch_dim=0, reduction=reduction)
            elbo = reduce_batch(elbo, batch_dim=0, reduction=reduction)

            other = list(other)
            for i in range(len(other)):
                other[i] = reduce_batch(other[i], batch_dim=0, reduction=reduction)

        return loss, elbo, *other

import logging

import torch


LOGGER = logging.getLogger()


def kl_divergence_mc(
    z: torch.Tensor, q_distrib: torch.distributions.Distribution, p_distrib: torch.distributions.Distribution
):
    """Elementwise Monte-Carlo estimation of KL between two distributions KL(q||p) (no reduction applied).

    Any number of dimensions works via broadcasting and correctly set `event_shape` (be careful).

    Args:
        z: Samples
        q_distrib: First distribution (Variational distribution)
        p_distrib: Second distribution

    Returns:
        tuple: Spatial KL divergence and log-likelihood of samples under q and under p (torch.Tensor)
    """
    q_logprob = q_distrib.log_prob(z)
    p_logprob = p_distrib.log_prob(z)
    kl_elementwise = q_logprob - p_logprob
    if kl_elementwise.isnan().any():
        LOGGER.warning(f"Encountered `nan` in KL divergence of shape {kl_elementwise.shape=}.")
    return kl_elementwise, q_logprob, p_logprob

import math

import numpy as np
import torch


class FreeNats:
    """Free bits as introduced in [1] but renamed to free nats because that's what it really is with log_e.

    In the paper they divide all latents Z into K groups. This implementation assumes a that each KL tensor passed
    to __call__ is one such group.

    The KL tensor may have more dimensions than the batch dimension, in which case the min_kl budget is distributed
    across those dimensions. E.g. if the shape is (32, 10), each of the 10 elements in the second dimension will get
    min_kl / 10 of the budget. If the shape is (32, 10, 10), each of the 10x10=100 elements will get min_kl / 100.

    [1] https://arxiv.org/pdf/1606.04934
    """

    def __init__(self, min_kl: float, shared_dims: int = None):
        self.min_kl = min_kl
        self.shared_dims = (shared_dims,) if isinstance(shared_dims, int) else shared_dims

    def __call__(self, kl: torch.Tensor) -> torch.Tensor:
        """
        Apply free nats over tensor. The free nats budget is distributed equally among dimensions.
        The returned free nats KL is equal to max(kl, freebits_per_dim, dim = >0)
        :param kl: KL of shape [batch size, *dimensions]
        :return:  free nats KL of shape [batch size, *dimensions]
        """
        if self.min_kl is None or self.min_kl == 0:
            return kl

        # equally divide free nats budget over the elements in shared_dims
        if self.shared_dims is not None:
            n_elements = math.prod([kl.shape[d] for d in self.shared_dims])
            min_kl_per_dim = self.min_kl / n_elements
        else:
            min_kl_per_dim = self.min_kl

        min_kl_per_dim = torch.tensor(min_kl_per_dim, dtype=kl.dtype, device=kl.device)
        freenats_kl = torch.maximum(kl, min_kl_per_dim)
        return freenats_kl

    def __repr__(self):
        return f"FreeNats({self.min_kl=})"


class FreeNatsCooldown:
    """Linear deterministic schedule for FreeNats."""

    def __init__(self, constant_epochs=200, cooldown_epochs=200, start_val=0.2, end_val=0):
        self.constant_epochs = constant_epochs
        self.cooldown_epochs = cooldown_epochs
        self.start_val = start_val
        self.end_val = start_val if constant_epochs == cooldown_epochs == 0 else end_val  # Start val if zero duration
        self.values = np.concatenate(
            [
                np.array([start_val] * constant_epochs),  # [start_val, start_val, ..., start_val]
                np.linspace(start_val, end_val, cooldown_epochs),  # [start_val, ..., end_val]
            ]
        )
        self.i_epoch = -1

    @property
    def is_done(self):
        return not self.i_epoch < len(self.values)

    def __iter__(self):
        return self

    def __next__(self):
        self.i_epoch += 1
        if self.is_done:
            return self.end_val
        return self.values[self.i_epoch]

    def __repr__(self):
        s = (
            f"FreeNatsCooldown(constant_epochs={self.constant_epochs}, cooldown_epochs={self.cooldown_epochs}, "
            f"start_val={self.start_val}, end_val={self.end_val})"
        )
        return s

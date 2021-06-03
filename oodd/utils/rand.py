import random

import torch
import numpy as np

import logging


LOGGER = logging.getLogger(name="oodd.utils.random")


def set_seed(seed):
    """Set the random number generation seed globally for `torch`, `numpy` and `random`"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    LOGGER.info("Set 'numpy', 'random' and 'torch' random seed to %s", seed)

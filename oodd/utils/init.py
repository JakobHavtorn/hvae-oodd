import math

import torch.nn as nn


def calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calculate_fan_by_mode(tensor, mode):
    mode = mode.lower()
    valid_modes = ["fan_in", "fan_out"]
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == "fan_in" else fan_out


def xavier_uniform_scale(tensor, gain=1):
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return bound


def xavier_normal_scale(tensor, gain=1):
    fan_in, fan_out = calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    return std


def kaiming_uniform_scale(tensor, mode="fan_in", gain=1):
    fan = calculate_fan_by_mode(tensor, mode)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return bound


def kaiming_normal_scale(tensor, mode="fan_in", gain=1):
    fan = calculate_fan_by_mode(tensor, mode)
    std = gain / math.sqrt(fan)
    return std


def get_activation_gain(activation, param=None):
    """Return the gain associated with the given activation according to

    Args:
        activation (torch.nn.Module): An activation function class (instantiated or not)

    Returns:
        float: The activation gain as returned by `nn.init.calculate_gain`
    """
    name = activation.__name__ if hasattr(activation, "__name__") else activation.__class__.__name__
    if activation is None:
        return 1
    elif name in ["LeakyReLU", "ELU", "Swish"]:
        return nn.init.calculate_gain("leaky_relu", param=param)
    name = name.lower()
    return nn.init.calculate_gain(name, param=param)

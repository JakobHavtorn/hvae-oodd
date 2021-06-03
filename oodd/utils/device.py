import os

from typing import Union

import torch


def get_device(idx: Union[int, None] = None):
    """Return the device to run on (cpu or cuda).

    If `CUDA_VISIBLE_DEVICES` is not set we assume that no devices are wanted and return the CPU.
    This is contrary to standard `torch.cuda.is_available()` behaviour

    If idx is specified, return the GPU corresponding to that index in the local scope.
    """
    if not torch.cuda.is_available() or "CUDA_VISIBLE_DEVICES" not in os.environ:
        return torch.device("cpu")

    if idx is None:
        return torch.device("cuda:0")

    local_device_indices = list(range(torch.cuda.device_count()))
    return torch.device(f"cuda:{local_device_indices[idx]}")


def test_gpu_functionality():
    """Returns `True` if a GPU is available and functionality is OK, otherwise raises an error"""
    # Set GPU as the device if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    print()

    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print("Memory Usage:")
        print("Allocated:", round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), "GB")
        print("Cached:   ", round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), "GB")
        print("CUDA version:", torch.version.cuda)
        return True
    else:
        # provoke an error
        torch.zeros(1).cuda()


if __name__ == '__main__':
    test_gpu_functionality()

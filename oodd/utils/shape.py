from typing import List, Tuple


def concatenate_shapes(shapes: List[Tuple[int]], axis: int):
    """Concatenate shapes along axis"""
    out = list(shapes[0])
    out[axis] = sum(list(s)[axis] for s in shapes)
    return tuple(out)


def flatten_sample_dim(tensor):
    """Flatten a tensor (S, B, *D) to (S * B, *D)."""
    return tensor.flatten(0, 1)


def elevate_sample_dim(tensor, n_samples):
    """Elevate a tensor's samples dimension from shape (B * S, *D) to (S, B, *D)"""
    batch_size = tensor.shape[0]
    assert (
        batch_size % n_samples == 0
    ), f'Number of samples does not divide the "batch size" ({batch_size}/{n_samples} = {batch_size/n_samples})'

    new_batch_size = batch_size // n_samples
    new_shape = (n_samples, new_batch_size, *(tensor.size()[1:]))

    return tensor.view(new_shape)


def copy_to_new_dim(x, n_copies, axis=0):
    """Create a view of a given tensor with a number of copies of the tensor created in a new dimension.

    By default, the new dimension is prepended and the returned shape is (C, *S) where *S refers to the original shape.

    Args:
        x (torch.Tensor): The tensor to copy.
        n_copies (int): The number of copies.
        axis (int): The location of the new axis in the returned tensor.
    """
    expanded_shape = (*x.shape[:axis], n_copies, *x.shape[axis:]) if axis != -1 else (*x.shape, n_copies)
    return x.unsqueeze(axis).expand(*expanded_shape)

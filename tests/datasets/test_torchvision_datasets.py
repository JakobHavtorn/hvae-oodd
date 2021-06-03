import torch
import pytest
import numpy as np

import oodd.datasets.torchvision_datasets as datasets

from oodd.constants import TEST_SPLIT, VAL_SPLIT


@pytest.mark.parametrize('dataset,n_examples,shape', [
    [datasets.MNISTQuantized, 10000, (1, 28, 28)],
    [datasets.FashionMNISTQuantized, 10000, (1, 28, 28)],
    [datasets.KMNISTQuantized, 10000, (1, 28, 28)],
    [datasets.CIFAR10Quantized, 10000, (3, 32, 32)],
    [datasets.SVHNQuantized, 531131, (3, 32, 32)],
])
def test_torchvision_dataset_quantized(dataset, n_examples, shape):
    d = dataset(split=VAL_SPLIT)

    assert d.size[0] == shape, "Image shapes"
    assert d.size[1] == torch.Size([])
    assert len(d) == n_examples, "Number of examples"

    examples = torch.stack([d[i][0] for i in range(1000)])

    assert examples.max() == 1
    assert examples.min() == 0
    assert len(examples.unique()) == 256


@pytest.mark.parametrize('dataset', [
    datasets.MNISTDequantized,
    datasets.FashionMNISTDequantized,
    datasets.KMNISTDequantized,
    datasets.CIFAR10Dequantized,
    datasets.SVHNDequantized,
])
def test_torchvision_dataset_dequantized(dataset):
    d = dataset(split=VAL_SPLIT)

    examples = torch.stack([d[i][0] for i in range(1000)])

    assert examples.max() > 0.99
    assert examples.min() < 0.01
    assert len(examples.unique()) > 256


@pytest.mark.parametrize('dataset', [
    datasets.MNISTBinarized,
    datasets.FashionMNISTBinarized,
    datasets.KMNISTBinarized,
])
def test_torchvision_dataset_binarized(dataset):
    d = dataset(split=VAL_SPLIT)

    examples = torch.stack([d[i][0] for i in range(1000)])

    assert examples.max() == 1
    assert examples.min() == 0
    assert len(examples.unique()) == 2

import torch

import oodd
import oodd.datasets.not_mnist as not_mnist


def test_init_notMNIST():
    d = not_mnist.notMNIST(train=False, root=oodd.constants.DATA_DIRECTORY, download=True)

    assert len(d) == 18724, "Number of examples"

    assert d.examples[0].max() == 255
    assert d.examples[0].min() == 0


def test_init_notMNISTQuantized():
    d = not_mnist.notMNISTQuantized(split="validation")

    assert d.size[0] == (1, 28, 28), "Image shapes"
    assert d.size[1] == torch.Size([])
    assert len(d) == 18724, "Number of examples"

    examples = torch.stack([d[i][0] for i in range(1000)])

    assert examples.max() == 1
    assert examples.min() == 0
    assert len(examples.unique()) == 256


def test_init_notMNISTDequantized():
    d = not_mnist.notMNISTDequantized(split="validation")

    assert d.size[0] == (1, 28, 28), "Image shapes"
    assert d.size[1] == torch.Size([])
    assert len(d) == 18724, "Number of examples"

    examples = torch.stack([d[i][0] for i in range(1000)])

    assert examples.max() > 0.99
    assert examples.min() < 0.01
    assert len(examples.unique()) > 256


def test_init_notMNISTBinarized():
    d = not_mnist.notMNISTBinarized(split="validation")

    assert d.size[0] == (1, 28, 28), "Image shapes"
    assert d.size[1] == torch.Size([])
    assert len(d) == 18724, "Number of examples"

    examples = torch.stack([d[i][0] for i in range(1000)])

    assert examples.max() == 1
    assert examples.min() == 0
    assert len(examples.unique()) == 2

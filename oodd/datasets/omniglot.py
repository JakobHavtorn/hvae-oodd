import argparse
import logging

import torchvision

from oodd.datasets import transforms
from oodd.datasets import BaseDataset
from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, DATA_DIRECTORY


LOGGER = logging.getLogger(__file__)


class OmniglotQuantized(BaseDataset):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    _data_source = torchvision.datasets.Omniglot
    _split_args = {TRAIN_SPLIT: {"background": True}, VAL_SPLIT: {"background": False}}

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    def __init__(
        self,
        split=TRAIN_SPLIT,
        root=DATA_DIRECTORY,
        transform=None,
        target_transform=None,
    ):

        super().__init__()

        transform = self.default_transform if transform is None else transform
        self.dataset = self._data_source(
            **self._split_args[split], root=root, transform=transform, target_transform=target_transform, download=True
        )

    @classmethod
    def get_argparser(cls):
        parser = argparse.ArgumentParser(description=cls.__name__)
        parser.add_argument("--root", type=str, default=DATA_DIRECTORY, help="Data storage location")
        return parser

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class OmniglotDequantized(OmniglotQuantized):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=2, min_val=0, max_val=1),  # Scale to [0, 2]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 3]
            transforms.Scale(a=0, b=1, min_val=0, max_val=3),  # Scale to [0, 1]
        ]
    )


OmniglotBinarized = OmniglotQuantized


class Omniglot28x28Quantized(OmniglotQuantized):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
        ]
    )


class Omniglot28x28Binarized(OmniglotQuantized):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            transforms.Binarize(resample=True),
        ]
    )


class Omniglot28x28Dequantized(OmniglotQuantized):
    """Omniglot dataset resized to 28x28 pixels (bilinear interpolation)"""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=2, min_val=0, max_val=1),  # Scale to [0, 2]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 3]
            transforms.Scale(a=0, b=1, min_val=0, max_val=3),  # Scale to [0, 1]
        ]
    )


class Omniglot28x28InvertedQuantized(OmniglotQuantized):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            transforms.InvertGrayScale(),
        ]
    )


class Omniglot28x28InvertedBinarized(OmniglotQuantized):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            transforms.InvertGrayScale(),
            transforms.Binarize(resample=True),
        ]
    )


class Omniglot28x28InvertedDequantized(OmniglotQuantized):
    """Omniglot dataset resized to 28x28 pixels (bilinear interpolation)"""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.NEAREST),
            torchvision.transforms.ToTensor(),
            transforms.InvertGrayScale(),
            transforms.Scale(a=0, b=2, min_val=0, max_val=1),  # Scale to [0, 2]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 3]
            transforms.Scale(a=0, b=1, min_val=0, max_val=3),  # Scale to [0, 1]
        ]
    )

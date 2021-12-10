import argparse
import logging
import os
import PIL
import tarfile

import tqdm
import numpy as np
import torch
import torch.utils.data as data
import torchvision

from urllib.request import urlretrieve

import oodd

from oodd.datasets import transforms
from oodd.datasets import BaseDataset
from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, DATA_DIRECTORY


LOGGER = logging.getLogger(__file__)


class ImageNet32(data.Dataset):
    """Base level ImageNet32 dataset"""

    train_filename = ""
    test_filename = ""

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        # self.root = os.path.join(self.root, "ImageNet32")

        if download:
            self.download()

        # self.dataset_tarfile = self.train_filename if train else self.test_filename
        #
        # self.examples, self.targets = load_dataset_from_file(self.dataset_tarfile)
        # self.targets = torch.LongTensor(self.targets)

        # self.shuffle(seed=19690720)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
        Returns:
            tuple: (image, target) where target is idx of the target class.
        """
        example, target = self.examples[idx], self.targets[idx]

        example = PIL.Image.fromarray(example.squeeze())  # 28x28 to PIL image

        if self.transform is not None:
            example = self.transform(example)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return example, target

    def download(self):
        pass

    def shuffle(self, seed):
        rng = np.random.default_rng(seed=seed)
        rand_idx = rng.permutation(list(range(len(self.examples))))
        self.examples = self.examples[rand_idx]
        self.targets = self.targets[rand_idx]

    def extract_to_folders(self):
        pass

    def __repr__(self):
        root, train, transform, target_transform, download = (
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        fmt_str = f"ImageNet32({root=}, {train=}, {transform=}, {target_transform=}, {download=})"
        return fmt_str

    def __len__(self):
        return len(self.examples)


class ImageNet32Quantized(BaseDataset):
    """ImageNet32 dataset serving quantized pixel values in [0, 1] (256 unique values)"""

    _data_source = ImageNet32
    _split_args = {TRAIN_SPLIT: {"train": True}, VAL_SPLIT: {"train": False}}

    default_transform = torchvision.transforms.ToTensor()

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


class ImageNet32Dequantized(ImageNet32Quantized):
    """ImageNet32 dataset serving dequantized pixel values in [0, 1] via 'x <- (x + u) / (255 + 1))'"""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
            transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        ]
    )


class ImageNet32Binarized(ImageNet32Quantized):
    """ImageNet32 dataset serving binarized pixel values in {0, 1} via """

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Binarize(resample=True),
        ]
    )

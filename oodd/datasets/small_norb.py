import argparse
import os
import os.path
import PIL
import struct
import math

import numpy as np
import torchvision
import torch.utils.data

from torchvision.datasets.utils import download_url, check_integrity

from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, DATA_DIRECTORY
from oodd.datasets import BaseDataset
from oodd.datasets import transforms


class SmallNORB(torch.utils.data.Dataset):
    """`small NORB <https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/>`_ Dataset.

    All images are uin8 with values in [0, 255].

    Args:
        root (string): Root directory of dataset
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    urls_train = [
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
            "smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz",
            "66054832f9accfe74a0f4c36a75bc0a2",
        ],
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
            "smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz",
            "23c8b86101fbf0904a000b43d3ed2fd9",
        ],
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
            "smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz",
            "51dee1210a742582ff607dfd94e332e3",
        ],
    ]
    urls_test = [
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
            "smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz",
            "e4ad715691ed5a3a5f138751a4ceb071",
        ],
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
            "smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz",
            "5aa791cd7e6016cf957ce9bdb93b8603",
        ],
        [
            "https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz",
            "smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat.gz",
            "a9454f3864d7fd4bb3ea7fc3eb84924e",
        ],
    ]

    train_data_file = ["smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat", "8138a0902307b32dfa0025a36dfa45ec"]
    train_labels_file = ["smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat", "fd5120d3f770ad57ebe620eb61a0b633"]
    train_info_file = ["smallnorb-5x46789x9x18x6x2x96x96-training-info.mat", "19faee774120001fc7e17980d6960451"]

    test_data_file = ["smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat", "e9920b7f7b2869a8f1a12e945b2c166c"]
    test_labels_file = ["smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat", "fd5120d3f770ad57ebe620eb61a0b633"]
    test_info_file = ["smallnorb-5x01235x9x18x6x2x96x96-testing-info.mat", "7c5b871cc69dcadec1bf6a18141f5edc"]

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.root = os.path.join(self.root, "smallNORB")

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted." + " You can use download=True to download it")

        if self.train:
            with open(os.path.join(self.root, self.train_data_file[0]), mode="rb") as f:
                self.train_data = self._parse_small_norb_data(f)
            with open(os.path.join(self.root, self.train_labels_file[0]), mode="rb") as f:
                self.train_labels = self._parse_small_norb_labels(f)
            with open(os.path.join(self.root, self.train_info_file[0]), mode="rb") as f:
                self.train_info = self._parse_small_norb_info(f)
        else:
            with open(os.path.join(self.root, self.test_data_file[0]), mode="rb") as f:
                self.test_data = self._parse_small_norb_data(f)
            with open(os.path.join(self.root, self.test_labels_file[0]), mode="rb") as f:
                self.test_labels = self._parse_small_norb_labels(f)
            with open(os.path.join(self.root, self.test_info_file[0]), mode="rb") as f:
                self.test_info = self._parse_small_norb_info(f)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        dindex = math.floor(index / 2)
        if self.train:
            img, target, info = self.train_data[index], self.train_labels[dindex], self.train_info[dindex]
        else:
            img, target, info = self.test_data[index], self.test_labels[dindex], self.test_info[dindex]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, info

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_integrity(self):
        root = self.root
        for fentry in [
            self.train_data_file,
            self.train_labels_file,
            self.train_info_file,
            self.test_data_file,
            self.test_labels_file,
            self.test_info_file,
        ]:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import gzip
        import shutil

        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        root = self.root

        for url in self.urls_train + self.urls_test:
            download_url(url[0], root, url[1], url[2])

            with gzip.open(os.path.join(root, url[1]), "rb") as f_in:
                with open(os.path.join(root, url[1][:-3]), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    def __repr__(self):
        fmt_str = "Dataset " + self.__class__.__name__ + "\n"
        fmt_str += "    Number of datapoints: {}\n".format(self.__len__())
        tmp = "train" if self.train is True else "test"
        fmt_str += "    Split: {}\n".format(tmp)
        fmt_str += "    Root Location: {}\n".format(self.root)
        tmp = "    Transforms (if any): "
        fmt_str += "{0}{1}\n".format(tmp, self.transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        tmp = "    Target Transforms (if any): "
        fmt_str += "{0}{1}".format(tmp, self.target_transform.__repr__().replace("\n", "\n" + " " * len(tmp)))
        return fmt_str

    def _parse_small_norb_header(self, file):
        magic = struct.unpack("<BBBB", file.read(4))
        ndims = struct.unpack("<i", file.read(4))[0]
        shape = []
        for _ in range(ndims):
            shape.append(struct.unpack("<i", file.read(4)))
        return {"magic_number": magic, "shape": shape}

    def _parse_small_norb_data(self, file):
        self._parse_small_norb_header(file)
        data = []
        buf = file.read(9216)
        while len(buf):
            data.append(PIL.Image.frombuffer("L", (96, 96), buf, "raw", "L", 0, 1))
            buf = file.read(9216)
        return data

    def _parse_small_norb_labels(self, file):
        self._parse_small_norb_header(file)
        file.read(8)
        data = []
        buf = file.read(4)
        while len(buf):
            data.append(struct.unpack("<i", buf)[0])
            buf = file.read(4)
        return data

    def _parse_small_norb_info(self, file):
        self._parse_small_norb_header(file)
        file.read(4)
        instance = []
        elevation = []
        azimuth = []
        lighting = []
        buf = file.read(4)
        while len(buf):
            instance.append(struct.unpack("<i", buf)[0])
            buf = file.read(4)
            elevation.append(struct.unpack("<i", buf)[0])
            buf = file.read(4)
            azimuth.append(struct.unpack("<i", buf)[0])
            buf = file.read(4)
            lighting.append(struct.unpack("<i", buf)[0])
            buf = file.read(4)
        return np.array([instance, elevation, azimuth, lighting]).transpose()


class SmallNORBQuantized(BaseDataset):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    _data_source = SmallNORB
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
        return self.dataset[idx][0], self.dataset[idx][1]


class SmallNORBBinarized(SmallNORBQuantized):
    """SmallNORB dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Binarize(resample=True),
        ]
    )


class SmallNORBDequantized(SmallNORBQuantized):
    """SmallNORB dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
            transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        ]
    )


class SmallNORB28x28Quantized(SmallNORBQuantized):
    """SmallNORB dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
        ]
    )


class SmallNORB28x28Binarized(SmallNORBQuantized):
    """SmallNORB dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            transforms.Binarize(resample=True),
        ]
    )


class SmallNORB28x28Dequantized(SmallNORBQuantized):
    """SmallNORB dataset resized to 28x28 pixels (bilinear interpolation)"""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=2, min_val=0, max_val=1),  # Scale to [0, 2]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 3]
            transforms.Scale(a=0, b=1, min_val=0, max_val=3),  # Scale to [0, 1]
        ]
    )


class SmallNORB28x28InvertedQuantized(SmallNORBQuantized):
    """Omniglot dataset including filtering and concatenation of train and test sets."""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((28, 28), interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
            torchvision.transforms.ToTensor(),
            transforms.InvertGrayScale(max_val=1),
        ]
    )

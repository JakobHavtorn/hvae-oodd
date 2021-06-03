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


def maybe_download(filename, data_root, url, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    destination_filename = os.path.join(data_root, filename)
    if not os.path.exists(destination_filename):
        print("Attempting to download:", filename)
        os.makedirs(os.path.dirname(destination_filename), exist_ok=True)
        filename, _ = urlretrieve(url + filename, destination_filename)  # , reporthook=download_progress_hook)
        print("\nDownload Complete!")

    statinfo = os.stat(destination_filename)
    if statinfo.st_size == expected_bytes:
        print("Found and verified", destination_filename)
    else:
        raise Exception("Failed to verify " + destination_filename + ". Can you get to it with a file browser?")

    return destination_filename


def load_dataset_from_file(dataset_tarfile):
    """
    Read previously extracted .npy files if available or read the images and targets from the tar file.
    Return the images and targets as numpy arrays.
    """
    dataset_directory = os.path.dirname(dataset_tarfile)
    dataset_tarfilename = os.path.basename(dataset_tarfile)
    dataset_tarfilename_no_extension = dataset_tarfilename.split(os.extsep)[0]  # Remove extension

    numpy_out_file_images = os.path.join(dataset_directory, dataset_tarfilename_no_extension) + ".npy"
    numpy_out_file_labels = os.path.join(dataset_directory, dataset_tarfilename_no_extension) + "_labels.npy"

    if not os.path.exists(numpy_out_file_images) or not os.path.exists(numpy_out_file_labels):
        # This is slower than loading from .npy files, which is why we we'd rather do that
        images, targets = read_tarfile(dataset_tarfile)
        print(f"Saving {numpy_out_file_images}")
        np.save(numpy_out_file_images, images)
        np.save(numpy_out_file_labels, targets)
    else:
        print(f"Loading {numpy_out_file_images}")
        images = np.load(numpy_out_file_images)
        targets = np.load(numpy_out_file_labels)

    return images, targets


def read_tarfile(dataset_tarfile):
    """Read the tarfile file by file and arrange the images in a single (B, 1, 28, 28) numpy array"""
    print(f"Reading image files in {dataset_tarfile}")
    tar = tarfile.open(dataset_tarfile, "r:gz")
    members = tar.getmembers()

    images = []
    targets = []
    for member in tqdm.tqdm(members):
        if member is not None and member.isfile():
            f = tar.extractfile(member)

            try:
                image = read_image(f)
            except PIL.UnidentifiedImageError:
                LOGGER.warning("Skipped file %s as it could not be read.", member.name)
                continue

            label = read_label(member)

            images.append(image)
            targets.append(label)

    images = np.stack(images)
    targets = np.array(targets)
    return images, targets


def read_image(tar_file):
    """Read a single image file in the tar into a numpy array"""
    image = PIL.Image.open(tar_file)
    image = np.array(image)
    image = image[np.newaxis, ...]
    return image


def read_label(tar_member):
    """Extract the label of an image by parsing the tarfile members name"""
    directory = os.path.dirname(tar_member.name)
    class_character = os.path.split(directory)[-1]  # A, B, C, ... as string
    class_label = ord(class_character) - ord("A")  # Converted to integer, 0, 1, 2, ...
    return class_label


def extract_dataset_tarfile(dataset_tarfilename):
    """
    Extract the dataset tar file creating the explicit folder structure of the dataset.
    The folder structure can then be read by 'read_dataset_folders'
    """
    tar = tarfile.open(dataset_tarfilename, "r:gz")
    directory = os.path.dirname(dataset_tarfilename)
    tar.extractall(directory)


def read_dataset_folders(root_folder):
    """Read the .png files extracted to folders.

    root_folder --> A --> img1.png
                      --> img2.png
                --> B --> img1.png
                      --> img2.png
         ...
                --> J --> img1.png
                      --> img2.png
    """
    print(root_folder)
    max_count = 0
    for (root, dirs, files) in os.walk(root_folder):
        for f in files:
            if f.endswith(".png"):
                max_count += 1

    print("Found %s files" % (max_count,))
    data = numpy.zeros((28, 28, max_count))
    targets = numpy.zeros((max_count,))
    count = 0
    for (root, dirs, files) in os.walk(root_folder):
        for f in files:
            if f.endswith(".png"):
                try:
                    img = PIL.Image.open(os.path.join(root, f))
                    data[:, :, count] = numpy.asarray(img)
                    root_folder = os.path.split(root)[-1]
                    assert len(root_folder) == 1
                    targets[count] = ord(root_folder) - ord("A")
                    count += 1
                except:
                    pass

    return data, targets


class notMNIST(data.Dataset):
    """Base level notMNIST dataset"""

    base_url = "http://yaroslavvb.com/upload/notMNIST/"
    train_filename = "notMNIST_large.tar.gz"
    test_filename = "notMNIST_small.tar.gz"
    train_expected_bytes = 247336696
    test_expected_bytes = 8458043

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.root = os.path.join(self.root, "notMNIST")

        if download:
            self.download()

        self.dataset_tarfile = self.train_filename if train else self.test_filename

        self.examples, self.targets = load_dataset_from_file(self.dataset_tarfile)
        self.targets = torch.LongTensor(self.targets)

        self.shuffle(seed=19690720)

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
        self.train_filename = maybe_download(self.train_filename, self.root, self.base_url, self.train_expected_bytes)
        self.test_filename = maybe_download(self.test_filename, self.root, self.base_url, self.test_expected_bytes)

    def shuffle(self, seed):
        rng = np.random.default_rng(seed=seed)
        rand_idx = rng.permutation(list(range(len(self.examples))))
        self.examples = self.examples[rand_idx]
        self.targets = self.targets[rand_idx]

    def extract_to_folders(self):
        extract_dataset_tarfile(self.train_filename)
        extract_dataset_tarfile(self.test_filename)

    def __repr__(self):
        root, train, transform, target_transform, download = (
            self.root,
            self.train,
            self.transform,
            self.target_transform,
            self.download,
        )
        fmt_str = f"notMNIST({root=}, {train=}, {transform=}, {target_transform=}, {download=})"
        return fmt_str

    def __len__(self):
        return len(self.examples)


class notMNISTQuantized(BaseDataset):
    """notMNIST dataset serving quantized pixel values in [0, 1] (256 unique values)"""

    _data_source = notMNIST
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


class notMNISTDequantized(notMNISTQuantized):
    """notMNIST dataset serving dequantized pixel values in [0, 1] via 'x <- (x + u) / (255 + 1))'"""

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Scale(a=0, b=255, min_val=0, max_val=1),  # Scale to [0, 1]
            transforms.Dequantize(),  # Add U(0, 1) noise, becomes [0, 256]
            transforms.Scale(a=0, b=1, min_val=0, max_val=256),  # Scale to [0, 1]
        ]
    )


class notMNISTBinarized(notMNISTQuantized):
    """notMNIST dataset serving binarized pixel values in {0, 1} via """

    default_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            transforms.Binarize(resample=True),
        ]
    )

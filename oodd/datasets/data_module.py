import argparse
import inspect
import logging
import os

from typing import List, Union, Dict, Any

import torch

from torch.utils.data import DataLoader, ConcatDataset

import oodd.datasets

from oodd.constants import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT
from oodd.utils.argparsing import json_file_or_json_unique_keys


LOGGER = logging.getLogger(name=__file__)


DATAMODULE_CONFIG_STR = "datamodule_config.pt"


def parse_dataset_argument(dataset_arg: Union[str, List[str], Dict[str, Dict[str, Any]]]):
    if isinstance(dataset_arg, str):
        return {dataset_arg: {}}
    if isinstance(dataset_arg, list):
        return {dataset_name: {} for dataset_name in dataset_arg}
    if isinstance(dataset_arg, dict):
        return {dataset_name: dataset_kwargs for dataset_name, dataset_kwargs in dataset_arg.items()}
    raise TypeError(f"Got dataset argument of type {type(dataset_arg)} but expected one of `str`, `list`, `dict`")


def get_dataset(dataset_name: str):
    """Split the dataset name key on hyphen in case some unique identifier was appended (more versions of dataset)"""
    dataset_name = dataset_name.split("-")
    dataset_name = dataset_name[0]
    return getattr(oodd.datasets, dataset_name)


class DataModule:
    """Module that serves datasets and dataloaders for training, validation and testing"""

    default_batch_size = 32
    default_data_workers = 2
    default_datasets = []

    test_batch_size_factor = 3

    def __init__(
        self,
        train_datasets: Union[str, List[str], Dict[str, Dict[str, Any]]] = default_datasets,
        val_datasets: Union[str, List[str], Dict[str, Dict[str, Any]]] = default_datasets,
        test_datasets: Union[str, List[str], Dict[str, Dict[str, Any]]] = default_datasets,
        batch_size: int = default_batch_size,
        test_batch_size: int = None,
        data_workers: int = default_data_workers,
    ):
        """A DataModule that serves several datasets for training, validation and testing.

        Datsets can be given either as
        1. A string with the 'class name'
        2. A list of string of several 'class names' (concatenated to single set for training sets)
        3. A dict of str of 'class names' and associated dict of str and kwargs for the individual datasets.
        In either case, the dataset argument is parsed to correspond to 3.

        The 'class name' should be that of a dataset in 'oodd.datasets'. If giving multiple different versions
        of the same dataset, the names of these should have an extra identifier appended

        When using this module, the potential additional 'kwargs' given via the dict input form take precedence over
        any dataset-specific arguments given via CLI arguments. This allows setting arguments that should apply to all
        datasets via the CLI (or the leave them default) while allowing overriding dataset-specific arguments as wanted.

        :param train_datasets: Training datasets
        :param val_datasets: Validation datasets
        :param test_datasets: Testing datasets
        :param batch_size: Batch size, defaults to default_batch_size
        :param data_workers: Number of parallel processes to use per dataset
        """
        self._batch_size = batch_size
        self._test_batch_size = self.test_batch_size_factor * batch_size if test_batch_size is None else test_batch_size
        self._data_workers = data_workers

        # Parse inputs
        train_datasets = parse_dataset_argument(train_datasets)
        val_datasets = parse_dataset_argument(val_datasets)
        test_datasets = parse_dataset_argument(test_datasets)

        self.config = dict(
            train_datasets=train_datasets,
            val_datasets=val_datasets,
            test_datasets=test_datasets,
            batch_size=batch_size,
            test_batch_size=test_batch_size,
            data_workers=data_workers,
        )

        self.train_datasets = {}
        self.val_datasets = {}
        self.test_datasets = {}

        # Build datasets and loaders
        self.add_datasets(train_datasets, val_datasets, test_datasets)

        # Define primary validation dataset
        self.primary_val_name = list(val_datasets.keys())[0]
        self.primary_val_dataset = self.val_datasets[self.primary_val_name]
        self.primary_val_loader = self.val_loaders[self.primary_val_name]

    def add_datasets(
        self,
        train_datasets: Dict[str, Dict[str, Any]] = {},
        val_datasets: Dict[str, Dict[str, Any]] = {},
        test_datasets: Dict[str, Dict[str, Any]] = {},
    ):
        """Build datasets for training, validation and test datasets"""
        for name, kwargs in train_datasets.items():
            if name in self.train_datasets:
                LOGGER.warning('Overwriting dataset %s', name)
            self.train_datasets[name], final_kwargs = DataModule._build_dataset(name, kwargs, TRAIN_SPLIT)
            self._update_config(dataset_group="train_datasets", dataset_name=name, kwargs=final_kwargs)

        for name, kwargs in val_datasets.items():
            if name in self.val_datasets:
                LOGGER.warning('Overwriting dataset %s', name)
            self.val_datasets[name], final_kwargs = DataModule._build_dataset(name, kwargs, VAL_SPLIT)
            self._update_config(dataset_group="val_datasets", dataset_name=name, kwargs=final_kwargs)

        for name, kwargs in test_datasets.items():
            if name in self.test_datasets:
                LOGGER.warning('Overwriting dataset %s', name)
            self.test_datasets[name], final_kwargs = DataModule._build_dataset(name, kwargs, TEST_SPLIT)
            self._update_config(dataset_group="test_datasets", dataset_name=name, kwargs=final_kwargs)

        # Concatenate the (potentially) multiple training datasets into one
        self.train_dataset = ConcatDataset(self.train_datasets.values()) if self.train_datasets else None
        
        self.recreate_dataloaders()

    def _update_config(self, dataset_group, dataset_name, kwargs):
        if dataset_name in self.config[dataset_group]:
            self.config[dataset_group][dataset_name].update(kwargs)
        else:
            self.config[dataset_group][dataset_name] = kwargs

    @staticmethod
    def _build_dataset(dataset_name: str, kwargs: dict, fallback_split: str):
        """Create a dataset in the defined split or use the split hash as the random seed

        Will not set the seed to the hash if the split is 'train' and the seed is set via the CLI.
        """
        # Get dataset and parse default arguments
        dataset = get_dataset(dataset_name)
        parser = dataset.get_argparser()
        args, unknown_args = parser.parse_known_args()

        signature = inspect.signature(dataset.__init__)
        dataset_missing_split_argument = "split" in signature.parameters and "split" not in kwargs
        dataset_missing_seed_argument = "seed" in signature.parameters and "seed" not in kwargs

        if dataset_missing_split_argument:
            args.split = fallback_split
        if dataset_missing_seed_argument and ("seed" not in args and fallback_split != "train"):
            args.seed = hash(fallback_split)

        args = vars(args)
        if kwargs:
            args.update(kwargs)

            # Print warning for overridden non-default valued CLI arguments if different from parsed args value
            non_default_override = [
                k for k in kwargs.keys() if args[k] != parser.get_default(k) and args[k] != kwargs[k]
            ]
            for k in non_default_override:
                s = f"Overriding non-default CLI argument '{k}={args[k]}' with value '{kwargs[k]}'"
                LOGGER.warning(s)

        args.pop('root', None)  # In case this is different for the specific install

        LOGGER.info("Creating dataset %s with args %s", dataset_name, args)
        dataset = dataset(**args)
        LOGGER.info("Created dataset %s", dataset)
        return dataset, args

    def recreate_dataloaders(self):
        self.train_loaders = {name: self._wrap_train_loader(dset) for name, dset in self.train_datasets.items()}
        self.val_loaders = {name: self._wrap_test_loader(dset) for name, dset in self.val_datasets.items()}
        self.test_loaders = {name: self._wrap_test_loader(dset) for name, dset in self.test_datasets.items()}
        self.train_loader = self._wrap_train_loader(self.train_dataset) if self.train_datasets else None

    def _wrap_train_loader(self, dataset):
        return self._wrap_dataloader(dataset, shuffle=True, batch_size=self.batch_size)

    def _wrap_test_loader(self, dataset):
        return self._wrap_dataloader(dataset, shuffle=False, batch_size=self.test_batch_size)

    def _wrap_dataloader(self, dataset, batch_size: int, shuffle: bool):
        dataloader = DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=self.data_workers,
            pin_memory=False,  # This cannot be True with persistent_workers True (at least not with non Tensor outputs)
            persistent_workers=True,  # This avoids reinstantiating the datasets (which would re-seed)
        )
        return dataloader

    @classmethod
    def get_argparser(cls, parents=[]):
        parser = argparse.ArgumentParser(description=cls.__name__, parents=parents, add_help=len(parents) == 0)
        parser.add_argument("--batch_size", type=int, default=cls.default_batch_size)
        parser.add_argument("--test_batch_size", type=int, default=None)
        parser.add_argument("--data_workers", type=int, default=cls.default_data_workers)
        parser.add_argument("--train_datasets", type=json_file_or_json_unique_keys, default=cls.default_datasets)
        parser.add_argument("--val_datasets", type=json_file_or_json_unique_keys, default=cls.default_datasets)
        parser.add_argument("--test_datasets", type=json_file_or_json_unique_keys, default=cls.default_datasets)
        return parser

    @property
    def batch_size(self):
        """Batch size used for training set loaders"""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        """Setting batch_size also updates the training set data loaders"""
        self._batch_size = batch_size
        self.train_loaders = {name: self._wrap_train_loader(dset) for name, dset in self.train_datasets.items()}
        self.train_loader = self._wrap_train_loader(self.train_dataset)

    @property
    def test_batch_size(self):
        """Batch size used for validation and test set loaders"""
        return self._test_batch_size

    @test_batch_size.setter
    def test_batch_size(self, test_batch_size):
        """Setting test_batch_size also updates the validation and test data loaders"""
        self._test_batch_size = test_batch_size
        self.val_loaders = {name: self._wrap_test_loader(dset) for name, dset in self.val_datasets.items()}
        self.test_loaders = {name: self._wrap_test_loader(dset) for name, dset in self.test_datasets.items()}

    @property
    def data_workers(self):
        return self._data_workers

    @data_workers.setter
    def data_workers(self, data_workers):
        self._data_workers = data_workers
        self.recreate_dataloaders()

    @property
    def size(self):
        return self.primary_val_dataset.size

    def save(self, path):
        torch.save(self.config, os.path.join(path, DATAMODULE_CONFIG_STR))

    @classmethod
    def load(cls, path, **override_kwargs):
        kwargs = torch.load(os.path.join(path, DATAMODULE_CONFIG_STR))
        kwargs.update(override_kwargs)
        return DataModule(**kwargs)

    def __repr__(self):
        tab = "    "
        s = "DataModule("
        s += f"\n{tab}batch_size={self.batch_size},"
        s += f"\n{tab}test_batch_size={self.test_batch_size},"
        s += f"\n{tab}data_workers={self.data_workers},"
        for attr in ["train_datasets", "val_datasets", "test_datasets"]:
            if len(getattr(self, attr).values()):
                s += f"\n{tab}{attr}=["
                s += f"\n{tab * 2}" + f"\n{tab * 2}".join(repr(dset) for dset in getattr(self, attr).values())
                s += f"\n{tab}],"
        s += "\n)"
        return s

"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import os
import logging

from collections import defaultdict
from typing import *

from tqdm import tqdm

import rich
import numpy as np
import torch

import oodd
import oodd.datasets
import oodd.evaluators
import oodd.models
import oodd.losses
import oodd.utils
from oodd.utils import reduce_to_batch

LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="/scratch/s193223/oodd/models/FashionMNIST-21-01-15-15-35-21.236574", help="model")
parser.add_argument("--save_dir", type=str, default="/scratch/s193223/oodd/results", help="directory to store scores in")

args = parser.parse_args()
rich.print(vars(args))

os.makedirs(args.save_dir, exist_ok=True)

FILE_NAME_SETTINGS_SPEC = f"k{args.n_latents_skip}-iw_elbo{args.iw_samples_elbo}-iw_lK{args.iw_samples_Lk}"

def get_save_path(name):
    name = name.replace(" ", "-")
    return f"{args.save_dir}/{name}"


def get_decode_from_p(n_latents, k=0, semantic_k=True):
    """
    k semantic out
    0 True     [False, False, False]
    1 True     [True, False, False]
    2 True     [True, True, False]
    0 False    [True, True, True]
    1 False    [False, True, True]
    2 False    [False, False, True]
    """
    if semantic_k:
        return [True] * k + [False] * (n_latents - k)

    return [False] * (k + 1) + [True] * (n_latents - k - 1)


def get_lengths(dataloaders):
    return [len(loader) for name, loader in dataloaders.items()]


def print_stats(llr, l, lk):
    llr_mean, llr_var, llr_std = np.mean(llr), np.var(llr), np.std(llr)
    l_mean, l_var, l_std = np.mean(l), np.var(l), np.std(l)
    lk_mean, lk_var, lk_std = np.mean(lk), np.var(lk), np.std(lk)
    s = f"  {l_mean=:8.3f},   {l_var=:8.3f},   {l_std=:8.3f}\n"
    s += f"{llr_mean=:8.3f}, {llr_var=:8.3f}, {llr_std=:8.3f}\n"
    s += f" {lk_mean=:8.3f},  {lk_var=:8.3f},  {lk_std=:8.3f}"
    print(s)


# Define checkpoints and load model
checkpoint = oodd.models.Checkpoint(path=args.model_dir)
checkpoint.load()
datamodule = checkpoint.datamodule
rich.print(datamodule)

# Add additional datasets to evaluation
TRAIN_DATASET_KEY = list(datamodule.train_datasets.keys())[0]
LOGGER.info("Train dataset %s", TRAIN_DATASET_KEY)

MAIN_DATASET_NAME = list(datamodule.train_datasets.keys())[0].strip("Binarized").strip("Quantized").strip("Dequantized")
LOGGER.info("Main dataset %s", MAIN_DATASET_NAME)

IN_DIST_DATASET = MAIN_DATASET_NAME + " test"
TRAIN_DATASET = MAIN_DATASET_NAME + " train"
LOGGER.info("Main in-distribution dataset %s", IN_DIST_DATASET)
if MAIN_DATASET_NAME in ["FashionMNIST", "MNIST"]:
    extra_val = dict(
        # notMNISTQuantized=dict(split='validation'),
        # Omniglot28x28Quantized=dict(split='validation'),
        # Omniglot28x28InvertedQuantized=dict(split='validation'),
        # SmallNORB28x28Quantized=dict(split='validation'),
        # SmallNORB28x28InvertedQuantized=dict(split='validation'),
        # KMNISTDequantized=dict(split='validation', dynamic=False),  # Effectively quantized
    )
    extra_test = {TRAIN_DATASET_KEY: dict(split="train", dynamic=False)}
elif MAIN_DATASET_NAME in ["CIFAR10", "SVHN"]:
    extra_val = dict(
        # CIFAR10DequantizedGrey=dict(split='test', preprocess='deterministic'),
        # CIFAR100Dequantized=dict(split='test', preprocess='deterministic'),
    )
    extra_test = {TRAIN_DATASET_KEY: dict(split="train", dynamic=False)}
else:
    raise ValueError(f"Unknown main dataset name {MAIN_DATASET_NAME}")

datamodule.add_datasets(val_datasets=extra_val, test_datasets=extra_test)
datamodule.data_workers = 4
datamodule.batch_size = 1
datamodule.test_batch_size = 1
LOGGER.info("%s", datamodule)

n_test_batches = get_lengths(datamodule.val_datasets) + get_lengths(datamodule.test_datasets)
N_EQUAL_EXAMPLES_CAP = min(n_test_batches)
assert N_EQUAL_EXAMPLES_CAP % 1 == 0, "Batch size must divide smallest dataset size"


N_EQUAL_EXAMPLES_CAP = min([args.n_eval_examples, N_EQUAL_EXAMPLES_CAP])
LOGGER.info("%s = %s", "N_EQUAL_EXAMPLES_CAP", N_EQUAL_EXAMPLES_CAP)

dataloaders = {(k + " test", v) for k, v in datamodule.val_loaders.items()}
dataloaders |= {(k + " train", v) for k, v in datamodule.test_loaders.items()}

entropies = defaultdict(list)

for dataset, dataloader in dataloaders:
    dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
    print(f"Evaluating {dataset}")

    n = 0
    for b, (x, _) in tqdm(enumerate(dataloader), total=N_EQUAL_EXAMPLES_CAP / 1):
        n += x.shape[0]
        print(x.shape, x.dtype)
        if n > N_EQUAL_EXAMPLES_CAP:
            LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP=}")
            break


# print likelihoods
for dataset in sorted(entropies.keys()):
    print("===============", dataset, "===============")
    print("mean entropy: ")
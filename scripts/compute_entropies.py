"""Script to evaluate the OODD scores (LLR and L>k) for a saved HVAE"""

import argparse
import os
import logging

from collections import defaultdict

from tqdm import tqdm

import rich
import numpy as np
import torch

import oodd.datasets
import oodd.utils

from skimage.filters.rank import entropy
from skimage.morphology import disk

LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str, default="FashionMNISTBinarized", help="model")
parser.add_argument("--n_eval_examples", type=int, default=float("inf"), help="cap on the number of examples to use")
parser.add_argument("--save_dir", type=str, default="/scratch/s193223/oodd/results", help="directory to store scores in")
parser = oodd.datasets.DataModule.get_argparser(parents=[parser])
args = parser.parse_args()
rich.print(vars(args))

os.makedirs(args.save_dir, exist_ok=True)

def get_save_path(name):
    name = name.replace(" ", "-")
    return f"{args.save_dir}/{name}"


def get_lengths(dataloaders):
    return [len(loader) for name, loader in dataloaders.items()]


# Data
datamodule = oodd.datasets.DataModule(
    batch_size=1,
    test_batch_size=1,
    data_workers=args.data_workers,
    train_datasets=args.train_datasets,
    val_datasets=args.val_datasets,
    test_datasets=args.test_datasets,
)

n_test_batches = get_lengths(datamodule.val_datasets) + get_lengths(datamodule.test_datasets)
N_EQUAL_EXAMPLES_CAP = min(n_test_batches)
assert N_EQUAL_EXAMPLES_CAP % 1 == 0, "Batch size must divide smallest dataset size"


N_EQUAL_EXAMPLES_CAP = min([args.n_eval_examples, N_EQUAL_EXAMPLES_CAP])
LOGGER.info("%s = %s", "N_EQUAL_EXAMPLES_CAP", N_EQUAL_EXAMPLES_CAP)

dataloaders = {(k + " test", v) for k, v in datamodule.val_loaders.items()}

complexities = defaultdict(list)


def complexity_metric(x):
    return np.mean(entropy(x, disk(3)))


for dataset, dataloader in dataloaders:
    dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
    print(f"Evaluating {dataset}")

    n = 0
    for b, (x, _) in tqdm(enumerate(dataloader), total=N_EQUAL_EXAMPLES_CAP / 1):
        n += x.shape[0]
        x = x[0]

        complexities[dataset].append(complexity_metric(x))

        if n > N_EQUAL_EXAMPLES_CAP:
            LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP=}")
            break


# print likelihoods
for dataset in sorted(complexities.keys()):
    print("===============", dataset, "===============")
    print("mean complexity: ", np.mean(complexities[dataset]))
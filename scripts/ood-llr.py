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


LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./models/FashionMNIST-21-01-15-15-35-21.236574", help="model")
parser.add_argument("--iw_samples_elbo", type=int, default=1, help="importances samples for regular ELBO")
parser.add_argument("--iw_samples_Lk", type=int, default=1, help="importances samples for L>k bound")
parser.add_argument("--n_eval_examples", type=int, default=float("inf"), help="cap on the number of examples to use")
parser.add_argument("--n_latents_skip", type=int, default=1, help="the value of k in the paper")
parser.add_argument("--batch_size", type=int, default=500, help="batch size for evaluation")
parser.add_argument("--device", type=str, default="auto", help="device to evaluate on")
parser.add_argument("--save_dir", type=str, default="./results", help="directory to store scores in")

args = parser.parse_args()
rich.print(vars(args))

os.makedirs(args.save_dir, exist_ok=True)
device = oodd.utils.get_device() if args.device == "auto" else torch.device(args.device)
LOGGER.info("Device %s", device)

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
model = checkpoint.model
model.eval()
criterion = oodd.losses.ELBO()
rich.print(datamodule)

# Add additional datasets to evaluation
TRAIN_DATASET_KEY = list(datamodule.train_datasets.keys())[0]
LOGGER.info("Train dataset %s", TRAIN_DATASET_KEY)

MAIN_DATASET_NAME = datamodule.primary_val_name.strip("Binarized").strip("Quantized").strip("Dequantized")
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
datamodule.batch_size = args.batch_size
datamodule.test_batch_size = args.batch_size
LOGGER.info("%s", datamodule)


n_test_batches = get_lengths(datamodule.val_datasets) + get_lengths(datamodule.test_datasets)
N_EQUAL_EXAMPLES_CAP = min(n_test_batches)
assert N_EQUAL_EXAMPLES_CAP % args.batch_size == 0, "Batch size must divide smallest dataset size"


N_EQUAL_EXAMPLES_CAP = min([args.n_eval_examples, N_EQUAL_EXAMPLES_CAP])
LOGGER.info("%s = %s", "N_EQUAL_EXAMPLES_CAP", N_EQUAL_EXAMPLES_CAP)

decode_from_p = get_decode_from_p(model.n_latents, k=args.n_latents_skip)

dataloaders = {(k + " test", v) for k, v in datamodule.val_loaders.items()}
dataloaders |= {(k + " train", v) for k, v in datamodule.test_loaders.items()}

scores = defaultdict(list)
elbos = defaultdict(list)
elbos_k = defaultdict(list)
with torch.no_grad():
    for dataset, dataloader in dataloaders:
        dataset = dataset.replace("Binarized", "").replace("Quantized", "").replace("Dequantized", "")
        print(f"Evaluating {dataset}")

        n = 0
        for b, (x, _) in tqdm(enumerate(dataloader), total=N_EQUAL_EXAMPLES_CAP / args.batch_size):
            x = x.to(device)

            n += x.shape[0]
            sample_elbos, sample_elbos_k = [], []

            # Regular ELBO
            for i in tqdm(range(args.iw_samples_elbo), leave=False):
                likelihood_data, stage_datas = model(x, decode_from_p=False, use_mode=False)
                kl_divergences = [
                    stage_data.loss.kl_elementwise
                    for stage_data in stage_datas
                    if stage_data.loss.kl_elementwise is not None
                ]
                loss, elbo, likelihood, kl_divergences = criterion(
                    likelihood_data.likelihood,
                    kl_divergences,
                    samples=1,
                    free_nats=0,
                    beta=1,
                    sample_reduction=None,
                    batch_reduction=None,
                )
                sample_elbos.append(elbo.detach())

            # L>k bound
            for i in tqdm(range(args.iw_samples_Lk), leave=False):
                likelihood_data_k, stage_datas_k = model(x, decode_from_p=decode_from_p, use_mode=decode_from_p)
                kl_divergences_k = [
                    stage_data.loss.kl_elementwise
                    for stage_data in stage_datas_k
                    if stage_data.loss.kl_elementwise is not None
                ]
                loss_k, elbo_k, likelihood_k, kl_divergences_k = criterion(
                    likelihood_data_k.likelihood,
                    kl_divergences_k,
                    samples=1,
                    free_nats=0,
                    beta=1,
                    sample_reduction=None,
                    batch_reduction=None,
                )
                sample_elbos_k.append(elbo_k.detach())

            sample_elbos = torch.stack(sample_elbos, axis=0)
            sample_elbos_k = torch.stack(sample_elbos_k, axis=0)

            sample_elbo = oodd.utils.log_sum_exp(sample_elbos, axis=0)
            sample_elbo_k = oodd.utils.log_sum_exp(sample_elbos_k, axis=0)

            score = sample_elbo - sample_elbo_k

            scores[dataset].extend(score.tolist())
            elbos[dataset].extend(sample_elbo.tolist())
            elbos_k[dataset].extend(sample_elbo_k.tolist())

            if n > N_EQUAL_EXAMPLES_CAP:
                LOGGER.warning(f"Skipping remaining iterations due to {N_EQUAL_EXAMPLES_CAP=}")
                break


# print likelihoods
for dataset in sorted(scores.keys()):
    print("===============", dataset, "===============")
    print_stats(scores[dataset], elbos[dataset], elbos_k[dataset])

# save scores
torch.save(scores, get_save_path(f"values-scores-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))
torch.save(elbos_k, get_save_path(f"values-elbos_k-{IN_DIST_DATASET}-{FILE_NAME_SETTINGS_SPEC}.pt"))

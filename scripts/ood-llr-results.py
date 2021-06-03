"""Script to compute metrics for the pre-computed scores (LLR and L>k) for a HVAE"""

import argparse
import os
import logging

from collections import defaultdict
from typing import *

import rich
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics

import oodd.evaluators
import oodd.models
import oodd.losses
import oodd.utils


LOGGER = logging.getLogger()


parser = argparse.ArgumentParser()
parser.add_argument("--source_dir", type=str, default="./results", help="directory from which to load scores")

args = parser.parse_args()
rich.print(vars(args))


# Helper methods
def collapse_multiclass_to_binary(y_true, zero_label=None):
    # Force the class index in zero_label to be zero and the others to collapse to 1
    zero_label_indices = y_true == zero_label
    y_true[zero_label_indices] = 0
    y_true[~zero_label_indices] = 1
    return y_true


def compute_roc_auc(y_true=None, y_score=None, zero_label=None):
    """Only binary"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    roc_auc = sklearn.metrics.roc_auc_score(y_true, y_score, average="macro")
    return roc_auc, fpr, tpr, thresholds


def compute_pr_auc(y_true=None, y_score=None, zero_label=None):
    """Only binary"""
    y_true = collapse_multiclass_to_binary(y_true, zero_label)
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_true, y_score)
    pr_auc = sklearn.metrics.average_precision_score(y_true, y_score, average="macro")
    return pr_auc, precision, recall, thresholds


def compute_roc_pr_metrics(y_true, y_score, reference_class):
    roc_auc, fpr, tpr, thresholds = compute_roc_auc(y_true=y_true, y_score=y_score, zero_label=reference_class)
    pr_auc, precision, recall, thresholds = compute_pr_auc(y_true=y_true, y_score=y_score, zero_label=reference_class)
    idx_where_tpr_is_eighty = np.where((tpr - 0.8 >= 0))[0][0]
    fpr80 = fpr[idx_where_tpr_is_eighty]
    return (roc_auc, fpr, tpr, thresholds), (pr_auc, precision, recall, thresholds), fpr80


def get_dataset(file_name):
    return " ".join(file_name.split("-")[2:4])


def get_iw(file_name):
    iw_elbo = int(file_name.split("-")[5][7:])
    iw_lK = int(file_name.split("-")[6][5:-3])
    return iw_elbo, iw_lK


def get_k(file_name):
    return int(file_name.split("-k")[-1].split("-")[0])


def load_data(files, negate_scores: bool = False):
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict))))
    for f in files:
        reference_dataset = get_dataset(f)
        iw_elbo, iw_elbo_k = get_iw(f)
        k = get_k(f)

        values = torch.load(os.path.join(args.source_dir, f))

        for test_dataset, values in values.items():
            values = np.array(values)
            data[reference_dataset][test_dataset][k][iw_elbo][iw_elbo_k] = values if not negate_scores else -values

    return data


def get_save_path(name):
    name = name.replace(" ", "-")
    return f"{args.source_dir}/{name}"


def write_text_file(filepath, string):
    with open(filepath, "w") as file_buffer:
        file_buffer.write(string)


def compute_results(score, score_name):
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(defaultdict))))
    for reference_dataset in score.keys():
        test_datasets = sorted(list(score[reference_dataset].keys()))
        s = f"========== {reference_dataset} (in-distribution) ==========\n"

        for test_dataset in test_datasets:
            k_values = sorted(list(score[reference_dataset][test_dataset].keys()))

            for k in k_values:
                iw_elbos = sorted(list(score[reference_dataset][test_dataset][k].keys()))

                for iw_elbo in iw_elbos:
                    iw_elbo_ks = sorted(list(score[reference_dataset][test_dataset][k][iw_elbo].keys()))

                    for iw_elbo_k in iw_elbo_ks:
                        reference_scores = score[reference_dataset][reference_dataset][k][iw_elbo][iw_elbo_k]
                        test_scores = score[reference_dataset][test_dataset][k][iw_elbo][iw_elbo_k]

                        # compute metrics
                        y_true = np.array([*[0] * len(reference_scores), *[1] * len(test_scores)])
                        y_score = np.concatenate([reference_scores, test_scores])

                        (
                            (roc_auc, fpr, tpr, thresholds),
                            (pr_auc, precision, recall, thresholds),
                            fpr80,
                        ) = compute_roc_pr_metrics(y_true=y_true, y_score=y_score, reference_class=0)

                        results[reference_dataset][test_dataset][k][iw_elbo][iw_elbo_k] = dict(
                            roc=dict(roc_auc=roc_auc, fpr=fpr, tpr=tpr, thresholds=thresholds),
                            pr=dict(pr_auc=pr_auc, precision=precision, recall=recall, thresholds=thresholds),
                            fpr80=fpr80,
                        )

                        s += f"{test_dataset:20s} | k={k:1d} | iw_elbo={iw_elbo:<4d} | iw_elbo_k={iw_elbo_k:<4d} | AUROC={roc_auc:6.4f}, AUPRC={pr_auc:6.4f}, FPR80={fpr80:6.4f}\n"

        print(s)
        f = f"results-{score_name}-{reference_dataset}.txt"
        write_text_file(get_save_path(f), s)

    return results


all_files = [f for f in os.listdir(args.source_dir) if f.endswith(".pt")]

all_scores = [f for f in all_files if "scores" in f]
all_elbo_k = [f for f in all_files if "elbos_k" in f]

scores = load_data(all_scores, negate_scores=False)
elbo_k = load_data(all_elbo_k, negate_scores=True)

rich.print(
    "[bold magenta]================================================ LLR ================================================[/]"
)
results_scores = compute_results(scores, score_name="llr")
rich.print(
    "[bold magenta]================================================ L>k ================================================[/]"
)
results_elbo_k = compute_results(elbo_k, score_name="elbo_k")

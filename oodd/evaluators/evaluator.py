import logging
import os

from collections import defaultdict
from typing import Dict, List, Union

import numpy as np
import torch

try:
    import wandb
except ImportError:
    pass


LOGGER = logging.getLogger(name=__file__)



def dd_dict():
    return dict()


def defaultdict_with_dict():
    return defaultdict(dd_dict)


class Evaluator:
    def __init__(
        self,
        primary_metric: str = None,
        primary_source: str = None,
        logger: logging.Logger = LOGGER,
        keep_on_device: bool = False,
        use_wandb: bool = True,
    ):
        """Evaluator that accumulates several metrics in a three-level data structure while keeping tensors on device.

        Hierarchy of the data structure for tracking metrics:
            Sources --> Series --> Metrics

        For example:
            MNISTBinarized --> likelihoods --> log p(x)
            MNISTBinarized --> likelihoods --> log p(x|z)
            MNISTBinarized --> divergences --> KL(q(z|x),p(z))
            FashionMNISTBinarized --> likelihoods --> log p(x)
            FashionMNISTBinarized --> likelihoods --> log p(x|z)
            FashionMNISTBinarized --> divergences --> KL(q(z|x),p(z))

        Series created inside a source are usually present in all other sources as well.
        Metrics created inside series must be unique to that series (no other series may hold that same metric)

        Series should be used to group sets of metrics that are comparable to each other in some way.
        Sources should be used to discern between series of metrics accumulated e.g. on different datasets.

        Args:
            primary_metric (str): Metric to use as primary indicator.
            primary_source (str): Data source to use as primary indicator. Defaults to the first one fed to `update()`.
            trains_logger (clearml.Logger): Trains logger for reporting experiment tracking information.
            tb_writer (tensorboard.SummaryWriter): SummaryWriter for Tensorboard experiment tracking.
            logger (logging.Logger): Logger to use when logging.
        """
        self._primary_source = primary_source
        self._primary_metric = primary_metric
        self.logger = logger
        self.keep_on_device = keep_on_device
        self.use_wandb = use_wandb

        self.reset()

    def reset(self):
        self.metrics = defaultdict(defaultdict_with_dict)

    def to(self, device):
        """Move all nested Tensor to `device`"""
        for source in self.metrics.keys():
            for series in self.metrics[source].keys():
                for metric_name in self.metrics[source][series].keys():
                    self.metrics[source][series][metric_name] = self.metrics[source][series][metric_name].to(device)

    @property
    def sources(self):
        """A list of source names"""
        return list(self.metrics.keys())

    @property
    def series(self):
        """A dictionary of a list of series names per source key"""
        return {source: list(self.metrics[source].keys()) for source in self.metrics.keys()}

    @property
    def primary_source(self):
        """Return the primary source (defaulting to the first source added to the Evaluator)"""
        if self._primary_source is None:
            return self.sources[0]
        return self._primary_source

    @property
    def primary_series(self, source=None):
        """Return the primary series being the series within the primary source that has the primary metric"""
        source = source if source is not None else self.primary_source
        primary_series = [
            series for series in self.series[source] if self.primary_metric in self.metrics[source][series]
        ]
        return primary_series[0]

    @property
    def primary_metric(self):
        if self._primary_metric is None:
            raise NotImplementedError("There is no default for the primary_metric")
        return self._primary_metric

    def get_primary_metric(self, source=None):
        """Return the series of the primary metric of the primary source assuming that it exists in only one series"""
        source = source if source is not None else self.primary_source
        return self.metrics[source][self.primary_series][self.primary_metric]

    def update(
        self,
        source: str = '',
        series: str = "default",
        metrics: Dict[str, Union[List, torch.Tensor]] = None,
    ):
        if metrics is not None:
            for metric_name, values in metrics.items():
                if not isinstance(values, torch.Tensor):
                    values = torch.Tensor(values)
                values = values.detach()

                if not self.keep_on_device:
                    values = values.to("cpu")

                if metric_name not in self.metrics[source][series]:
                    self.metrics[source][series][metric_name] = values
                else:
                    # Concatenate to keep all values
                    self.metrics[source][series][metric_name] = torch.cat(
                        [self.metrics[source][series][metric_name], values]
                    )

    def save(self, path, idx=None):
        """Save evaluators settings except loggers as they can't be pickled"""
        kwargs = dict(
            primary_metric=self._primary_metric, primary_source=self._primary_source, keep_on_device=self.keep_on_device
        )
        store_dict = dict(kwargs=kwargs, metrics=self.metrics)
        name = "evaluator.pt" if idx is None else f"evaluator_{idx}.pt"
        torch.save(store_dict, os.path.join(path, name))

    @classmethod
    def load(cls, path, idx=None):
        name = "evaluator.pt" if idx is None else f"evaluator_{idx}.pt"
        store_dict = torch.load(os.path.join(path, name))
        evaluator = cls(**store_dict["kwargs"])
        evaluator.metrics = store_dict["metrics"]
        return evaluator

    def report(self, iteration: int):
        """E.g. report to some experiment framework"""
        if self.use_wandb:
            log_dict = dict()
            for source in self.sources:
                s = f"{source}"
                for series in self.series[source]:
                    ss = s + f".{series}"
                    for metric_name in self.metrics[source][series]:
                        name = ss + f".{metric_name}"
                        value = self.metrics[source][series][metric_name].mean()
                        log_dict[name] = value
            log_dict = {k: v.item() for k, v in log_dict.items()}
            wandb.log(log_dict, step=iteration)

    def string(self, iteration: int = None, iteration_name: str = "Epoch"):
        pre = ''
        if iteration_name:
            pre += f"{iteration_name}"
        if iteration is not None:
            pre += f" {iteration:<4}"
        pre_len = len(pre)
        pre_len_whitespace = " " * pre_len

        source_strings = []
        source_max_chars = max(len(source) for source in self.sources)
        series_max_chars = max(len(series) for source in self.sources for series in self.series[source])
        for source in self.sources:
            s = [f"{source:<{source_max_chars}s}"]

            for series in self.series[source]:
                ss = [f"{series:<{series_max_chars}s}"]
                for metric_name in self.metrics[source][series]:
                    ss += [f"{metric_name} {self.metrics[source][series][metric_name].mean():.2E}"]
                ss = " | ".join(ss)
                s.append(f"\n{pre_len_whitespace}{ss}")

            s = ''.join(s)
            source_strings.append(s)

        source_strings = [source_strings[0]] + [s for s in source_strings[1:]]
        source_strings = "\n".join(source_strings)

        return pre + '\n' + source_strings

    def log(self, iteration: int = "", iteration_name="Epoch"):
        self.logger.info(self.string(iteration, iteration_name=iteration_name))

    def print(self, iteration: int = "", iteration_name="Epoch"):
        print(self.string(iteration, iteration_name=iteration_name))

    def get_classes_and_scores_per_source(self, series=None, metric_name=None, reverse_label_order=False):
        """
        Loop over sources and return per datapoint, the model score (as defined by 'series' and 'metric_nane', defaults 
        to primary) and the label corresponding to the source on which the metric was obtained.
        """
        # default to primary series and metric
        series = series if series is not None else self.primary_series
        metric_name = metric_name if metric_name is not None else self.primary_metric

        # all sources that contains the series (non-empty)
        sources = [source for source in self.sources if self.metrics[source][series].get(metric_name) is not None]

        if reverse_label_order:
            classes = {source: i for i, source in enumerate(reversed(sources))}
        else:
            classes = {source: i for i, source in enumerate(sources)}

        y_true, y_score = [], []
        for source in sources:
            scores = self.metrics[source][series][metric_name].tolist()
            y_score.extend(scores)
            y_true.extend([classes[source]] * len(scores))

        return np.array(y_true), np.array(y_score), classes

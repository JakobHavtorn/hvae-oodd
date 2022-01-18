# hvae-oodd

<a href="https://doi.org/10.5281/zenodo.5873322"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.5873322.svg" alt="DOI"></a>

Official source code repository for the ICML 2021 paper "Hierarchical VAEs Know What They Don't Know"

arXiv paper link: https://arxiv.org/abs/2102.08248.

Source code builds upon https://github.com/vlievin/biva-pytorch.

![Reconstructions of FashionMNIST and MNIST examples by latent variable trained on FashionMNIST](https://github.com/JakobHavtorn/hvae-oodd/blob/main/assets/reconstructions.png)


## Install

```
conda deactivate
conda env remove -n oodd -y
conda create -y -n oodd python==3.8.10
conda activate oodd
conda install -y -c pytorch pytorch torchvision torchaudio cudatoolkit=11.3
pip install -r requirements.txt
pip install --editable .
```

Make sure that the CUDA version used by `torch` corresponds to the one on the system.


## Test

To run package tests:

> `pytest -v --cov --cov-report=term tests`

Please run these to ensure that the code works as expected on your system.


## Train a model

The below commands will train a HVAE on binarized FashionMNIST and dequantized CIFAR10.
To train on binarized MNIST or dequantized SVHN, just switch the dataset names.

The first time any of the commands are run, we will first download the source data via `torchvision`.

By default, checkpoint files will be stored in `./models`.


**FashionMNIST/MNIST**

```bash
python scripts/dvae_run.py \
--epochs 1000 \
--batch_size 128 \
--free_nats 2 \
--free_nats_epochs 400 \
--warmup_epochs 200 \
--test_every 10 \
--train_datasets \
'{
    "FashionMNISTBinarized": {"dynamic": true, "split": "train"}
}' \
--val_datasets \
'{
    "FashionMNISTBinarized": {"dynamic": false, "split": "validation"},
    "MNISTBinarized": {"dynamic": false, "split": "validation"}
}' \
--model VAE \
--likelihood BernoulliLikelihoodConv2d \
--config_deterministic \
'[
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false}
    ]
]' \
--config_stochastic \
'[
    {"block": "GaussianConv2d", "latent_features": 8, "weightnorm": true},
    {"block": "GaussianDense", "latent_features": 16, "weightnorm": true},
    {"block": "GaussianDense", "latent_features": 8, "weightnorm": true}
]'
```


**CIFAR10/SVHN**

```bash
python scripts/dvae_run.py \
--epochs 2000 \
--batch_size 128 \
--free_nats 2 \
--free_nats_epochs 400 \
--warmup_epochs 200 \
--test_every 10 \
--train_datasets '{ "CIFAR10Dequantized": {"dynamic": true, "split": "train"}}' \
--val_datasets \
'{
    "CIFAR10Dequantized": {"dynamic": false, "split": "validation"},
    "SVHNDequantized": {"dynamic": false, "split": "validation"}
}' \
--model VAE \
--likelihood DiscretizedLogisticMixLikelihoodConv2d \
--config_deterministic \
'[
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 5, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ]
]' \
--config_stochastic \
'[
    {"block": "GaussianConv2d", "latent_features": 128, "weightnorm": true},
    {"block": "GaussianConv2d", "latent_features": 64, "weightnorm": true},
    {"block": "GaussianDense", "latent_features": 32, "weightnorm": true}
]'
```


## Evaluate models

Evaluation without importance sampling can be run on CPU but we recommend using a single GPU. 

The `batch_size` can be tuned to avoid out-of-memory errors.

1. Compute OODD scores  (LLR>k and L>k):
   > `python scripts/ood-llr.py --model models/FashionMNISTBinarized-YYYY-MM-DD-hh-mm-ss.milli/ --iw_samples_elbo 1 --n_latents_skip 0`

   > `python scripts/ood-llr.py --model models/FashionMNISTBinarized-YYYY-MM-DD-hh-mm-ss.milli/ --iw_samples_elbo 1 --n_latents_skip 1`

   > `python scripts/ood-llr.py --model models/FashionMNISTBinarized-YYYY-MM-DD-hh-mm-ss.milli/ --iw_samples_elbo 1 --n_latents_skip 2`

   > `python scripts/ood-llr.py --model models/FashionMNISTBinarized-YYYY-MM-DD-hh-mm-ss.milli/ --iw_samples_elbo 1000 --n_latents_skip 1`

   > `python scripts/ood-llr.py --model models/FashionMNISTBinarized-YYYY-MM-DD-hh-mm-ss.milli/ --iw_samples_elbo 1000 --n_latents_skip 2`

2. Compute OODD results
    > `python scripts/ood-llr-results.py`


## Citation

```
@InProceedings{pmlr-v139-havtorn21a,
  title = {Hierarchical VAEs Know What They Don’t Know},
  author = {Havtorn, Jakob D. Drachmann and Frellsen, Jes and Hauberg, Soren and Maal{\o}e, Lars},
  booktitle = {Proceedings of the 38th International Conference on Machine Learning},
  pages = {4117--4128},
  year = {2021},
  editor = {Meila, Marina and Zhang, Tong},
  volume = {139},
  series = {Proceedings of Machine Learning Research},
  month = {18--24 Jul},
  publisher = {PMLR},
  pdf = {http://proceedings.mlr.press/v139/havtorn21a/havtorn21a.pdf},
  url = {http://proceedings.mlr.press/v139/havtorn21a.html},
  abstract = {Deep generative models have been demonstrated as state-of-the-art density estimators. Yet, recent work has found that they often assign a higher likelihood to data from outside the training distribution. This seemingly paradoxical behavior has caused concerns over the quality of the attained density estimates. In the context of hierarchical variational autoencoders, we provide evidence to explain this behavior by out-of-distribution data having in-distribution low-level features. We argue that this is both expected and desirable behavior. With this insight in hand, we develop a fast, scalable and fully unsupervised likelihood-ratio score for OOD detection that requires data to be in-distribution across all feature-levels. We benchmark the method on a vast set of data and model combinations and achieve state-of-the-art results on out-of-distribution detection.}
}
```

# hvae-oodd

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
conda install -y -c pytorch python pytorch torchvision cudatoolkit=10.2
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
--epochs 1000 \
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
        {"block": "ResBlockConv2d", "out_channels": 32, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 32, "kernel_size": 5, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 5, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 64, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 128, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ],
    [
        {"block": "ResBlockConv2d", "out_channels": 128, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 128, "kernel_size": 3, "stride": 1, "weightnorm": true, "gated": false},
        {"block": "ResBlockConv2d", "out_channels": 256, "kernel_size": 3, "stride": 2, "weightnorm": true, "gated": false}
    ]
]' \
--config_stochastic \
'[
    {"block": "GaussianConv2d", "latent_features": 8, "weightnorm": true},
    {"block": "GaussianConv2d", "latent_features": 16, "weightnorm": true},
    {"block": "GaussianConv2d", "latent_features": 32, "weightnorm": true}
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

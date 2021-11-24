# FashionMNIST model
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/FashionMNISTBinarized-2021-11-23-17-12-45.552811/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/FashionMNISTBinarized-2021-11-23-17-12-45.552811/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/FashionMNISTBinarized-2021-11-23-17-12-45.552811/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model /scratch/s193223/oodd/models/FashionMNISTBinarized-2021-11-23-17-12-45.552811/ --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/FashionMNISTBinarized-2021-11-23-17-12-45.552811/ --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

# MNIST model
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/MNISTBinarized-2021-11-23-17-14-19.091676/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/MNISTBinarized-2021-11-23-17-14-19.091676/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/MNISTBinarized-2021-11-23-17-14-19.091676/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model /scratch/s193223/oodd/models/MNISTBinarized-2021-11-23-17-14-19.091676/ --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/MNISTBinarized-2021-11-23-17-14-19.091676/ --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

# CIFAR10 model
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/CIFAR10Dequantized-2021-11-23-16-59-48.686207/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/CIFAR10Dequantized-2021-11-23-16-59-48.686207/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/CIFAR10Dequantized-2021-11-23-16-59-48.686207/ --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model /scratch/s193223/oodd/models/CIFAR10Dequantized-2021-11-23-16-59-48.686207/ --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/CIFAR10Dequantized-2021-11-23-16-59-48.686207/ --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

# SVHN model
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/SVHNDequantized-2021-11-23-17-16-27.156307 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/SVHNDequantized-2021-11-23-17-16-27.156307 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/SVHNDequantized-2021-11-23-17-16-27.156307 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model /scratch/s193223/oodd/models/SVHNDequantized-2021-11-23-17-16-27.156307 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model /scratch/s193223/oodd/models/SVHNDequantized-2021-11-23-17-16-27.156307 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

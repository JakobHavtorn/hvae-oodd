# FashionMNIST model
python scripts/ood-llr.py --model ./models/FashionMNISTBinarized-2021-01-15-15-35-21.236574 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model ./models/FashionMNISTBinarized-2021-01-15-15-35-21.236574 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/FashionMNISTBinarized-2021-01-15-15-35-21.236574 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model ./models/FashionMNISTBinarized-2021-01-15-15-35-21.236574 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/FashionMNISTBinarized-2021-01-15-15-35-21.236574 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

# MNIST model
python scripts/ood-llr.py --model ./models/MNISTBinarized-2021-01-15-15-37-24.759273 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model ./models/MNISTBinarized-2021-01-15-15-37-24.759273 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/MNISTBinarized-2021-01-15-15-37-24.759273 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model ./models/MNISTBinarized-2021-01-15-15-37-24.759273 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/MNISTBinarized-2021-01-15-15-37-24.759273 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

# CIFAR10 model
python scripts/ood-llr.py --model ./models/CIFAR10Dequantized-2021-01-15-15-40-51.849542 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model ./models/CIFAR10Dequantized-2021-01-15-15-40-51.849542 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/CIFAR10Dequantized-2021-01-15-15-40-51.849542 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model ./models/CIFAR10Dequantized-2021-01-15-15-40-51.849542 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/CIFAR10Dequantized-2021-01-15-15-40-51.849542 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

# SVHN model
python scripts/ood-llr.py --model ./models/SVHNDequantized-2021-01-15-15-42-53.118353 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 0
python scripts/ood-llr.py --model ./models/SVHNDequantized-2021-01-15-15-42-53.118353 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/SVHNDequantized-2021-01-15-15-42-53.118353 --iw_samples_elbo 1 --iw_samples_Lk 1 --n_latents_skip 2

python scripts/ood-llr.py --model ./models/SVHNDequantized-2021-01-15-15-42-53.118353 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 1
python scripts/ood-llr.py --model ./models/SVHNDequantized-2021-01-15-15-42-53.118353 --iw_samples_elbo 1000 --iw_samples_Lk 1 --n_latents_skip 2

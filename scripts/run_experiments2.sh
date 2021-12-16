GPU=2
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/bnw_dequantized.json"

echo "Running experiments on GPU $GPU"

#         'FashionMNISTDequantized': {'split': 'validation', 'dynamic': False},
#        'MNISTDequantized': {'split': 'validation', 'dynamic': False},
#        'notMNISTDequantized': {'split': 'validation'},
#        'Omniglot28x28Dequantized': {'split': 'validation'},
#        'Omniglot28x28InvertedDequantized': {'split': 'validation'},
# not doing small norb as it's bad binarized
#        'SmallNORB28x28Dequantized': {'split': 'validation'},
#        'KMNISTDequantized': {'split': 'validation', 'dynamic': False}

CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "FashionMNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "MNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "KMNISTDequantized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic.json





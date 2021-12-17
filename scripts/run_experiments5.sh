GPU=5
LIKELIHOOD="DiscretizedLogisticMixLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/cifar_topics.json"


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "CIFAR10DequantizedAnimals": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic.json


CUDA_VISIBLE_DEVICES=$GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "CIFAR10DequantizedTransportation": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic.json





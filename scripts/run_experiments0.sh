GPU=0
LIKELIHOOD="BernoulliLikelihoodConv2d"
VAL_DATA="scripts/configs/val_datasets/binarized.json"

echo "Running experiments on GPU $GPU"


CUDA_VISIBLE_DEVICES=GPU python scripts/dvae_run.py \
                               --seed 1 \
                               --train_datasets \
                               '{
                                   "FashionMNISTBinarized": {"dynamic": true, "split": "train"}
                               }' \
                               --val_datasets $VAL_DATA \
                               --likelihood $LIKELIHOOD \
                               --config_deterministic scripts/configs/default_model/config_deterministic.json \
                               --config_stochastic scripts/configs/default_model/config_stochastic.json


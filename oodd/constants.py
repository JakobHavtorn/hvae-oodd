import logging
import os


# Logging
LOG_FORMAT = "%(asctime)-15s - %(module)-20s - %(levelname)-7s | %(message)s"
LOG_LEVEL = os.getenv("HVAE_OODD_LOG_LEVEL", "INFO")
logging.basicConfig(format=LOG_FORMAT, level=logging.getLevelName(LOG_LEVEL))

# Split keys used throughout dataset handling
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"

# Metric keys
LOG_P_X = "log p(x)"
LOG_P_X_Z = "log p(x|z)"
KL_DIVERGENCE = "KL(q(z|x), p(z)"

ROOT_PATH = __file__.replace("/oodd/constants.py", "")
DATA_DIRECTORY = os.path.join(ROOT_PATH, "data")

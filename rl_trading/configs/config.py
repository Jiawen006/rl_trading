"""time module will be used to name file directory"""
from datetime import datetime

import numpy as np

WINDOW_LENGTH = 90
DATA_FOLDER = "datasets/"
TRAINING_DATA_NAME = "datasets/train_processed.csv"
TRAINING_DATA_FOLDER = "datasets/train"
TEST_DATA_NAME = "datasets/test_processed.csv"
TEST_DATA_FOLDER = "datasets/test"
TURBULENCE_START = 50
INDICATORS = ["macd", "rsi", "cci", "adx"]
INITIAL_AMOUNT = 1000000
HMAX = 100

now = datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
RESULT = f"results/{now}"

# training hyperparameter
A2CTIMESTEPS = 30000
PPOTIMESTEPS = 30000
DDPG_TIMESTEPS = 20000


A2C_PARAMS = {
    "n_steps": 10,
    "ent_coef": 0.001,
    "learning_rate": 0.0002,
    "verbose": 0,
}
PPO_PARAMS = {
    "n_steps": 128,
    "ent_coef": 0.001,
    "learning_rate": 0.00003,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.0003}

# Environment hyperparameter
STATE_SPACE = 91
ACTION_SPACE = 10
STOCK_DIM = 10
HMAX = 100
TRANSACTION_COST_PCT = 0.2
PROFIT_REWARD_SCALING = 1e-9
LOSS_REWARD_SCALING = 2e-9

SERIES_WEIGHT = np.array([10, 200, 1, 10, 1, 100, 0.1, 100, 1, 5])
ORDER_COEFFICIENT = 10 * SERIES_WEIGHT
SHORT_THRESHOLD = -1000 * SERIES_WEIGHT
BANKRUPT_PENALTY = -1000000

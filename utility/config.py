"""time module will be used to name file directory"""
from datetime import datetime

import numpy as np

WINDOW_LENGTH = 90
DATA_FOLDER = "data/"
TRAINING_DATA_NAME = "data/train_processed.csv"
TRAINING_DATA_FOLDER = "data/train"
TEST_DATA_NAME = "data/test_processed.csv"
TEST_DATA_FOLDER = "data/test"
TURBULENCE_START = 50
INDICATORS = ["macd", "rsi", "cci", "adx"]
INITIAL_AMOUNT = 1000000
HMAX = 100

now = datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
RESULT = f"results/{now}"

A2CTIMESTEPS = 30000
PPOTIMESTEPS = 20000
DDPG_TIMESTEPS = 10000
A2CTIMESTEPS = 300
PPOTIMESTEPS = 200
DDPG_TIMESTE = 100


A2C_PARAMS = {
    "n_steps": 20,
    "ent_coef": 0.003,
    "learning_rate": 0.0001,
    "verbose": 0,
}
PPO_PARAMS = {
    "n_steps": 1024,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}

# Environment hyperparameter
TRANSACTION_COST_PCT = 0.2
PROFIT_REWARD_SCALING = 1e-9
LOSS_REWARD_SCALING = 2e-9

SERIES_WEIGHT = np.array([10, 200, 1, 10, 1, 100, 0.1, 100, 1, 5])
ORDER_COEFFICIENT = 10 * SERIES_WEIGHT
SHORT_THRESHOLD = -1000 * SERIES_WEIGHT
BANKRUPT_PENALTY = -1000000

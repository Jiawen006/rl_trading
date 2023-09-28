from datetime import datetime

import numpy as np

WINDOW_LENGTH = 90
DATA_FOLDER = "DATA/"
TRAINING_DATA_NAME = "DATA/train_processed.csv"
TRAINING_DATA_FOLDER = "DATA/Train"
TEST_DATA_NAME = "DATA/test_processed.csv"
TEST_DATA_FOLDER = "DATA/Test"
TURBULENCE_START = 50
INDICATORS = ["macd", "rsi", "cci", "adx"]
INITIAL_AMOUNT = 1000000
HMAX = 100

now = datetime.now()
TRAINED_MODEL_DIR = f"trained_models/{now}"
RESULT = f"results/{now}"

a2ctimesteps = 30000
ppotimesteps = 20000
ddpgtimestep = 10000

A2C_PARAMS = {
    "n_steps": 20,
    "ent_coef": 0.003,
    "learning_rate": 0.0001,
    "verbose": 0,
}
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0025,
    "batch_size": 64,
}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.0001}

series_weight = np.array([10, 200, 1, 10, 1, 100, 0.1, 100, 1, 5])
order_coefficient = 10 * series_weight
short_threshold = -1000 * series_weight
banrupt_penalty = -1000000

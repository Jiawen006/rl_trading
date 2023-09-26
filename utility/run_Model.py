import os

import numpy as np
import pandas as pd
from stable_baselines3 import A2C

from config import config
from env.environment_trade import StockEnvTrade
from utility import preprocessor
from utility.ensemble import ensemble_strategy
from utility.train import run_train


def run():
    data_folder_path = config.DATA_FOLDER
    train_data, test_data = data_load(folder_name=data_folder_path)
    print("Data loading is complete.")

    # model = train_model(train_data)
    # print("Data training is complete")

    model = A2C.load("trained_models/a2c_train.zip")

    ensemble_strategy(model=model, data=test_data, window_length=config.WINDOW_LENGTH)


def data_load(folder_name):
    # check whether folder exist
    if not os.path.exists(folder_name):
        raise Exception("DATA Folder not found.")
    # check whether processed data exist
    if os.path.exists(config.TRAINING_DATA_NAME):
        train_data = pd.read_csv(config.TRAINING_DATA_NAME, index_col=0)
    else:
        train_data = preprocessor.preprocess_pipeline(config.TRAINING_DATA_FOLDER)
        train_data.to_csv(config.TRAINING_DATA_NAME, index=True)

    if os.path.exists(config.TEST_DATA_NAME):
        test_data = pd.read_csv(config.TEST_DATA_NAME, index_col=0)
    else:
        test_data = preprocessor.preprocess_pipeline(config.TEST_DATA_FOLDER)
        test_data.to_csv(config.TEST_DATA_NAME, index=True)

    return train_data, test_data


def train_model(data):
    # this method train the model before trading
    model = run_train(data)
    return model

import os

import pandas as pd
import preprocessor
from stable_baselines3.common.vec_env import DummyVecEnv
from train import *

from env.environment_train import StockEnvTrain


def run():
    data_folder_path = config.DATA_FOLDER
    train_data, test_data = data_load(folder_name=data_folder_path)
    print("Data loading is complete.")

    model = train_model(train_data)

    # to do
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
    model = ""
    env_train = DummyVecEnv([lambda: StockEnvTrain(data)])
    run_train(env_train)
    return model


def ensemble_strategy(model, data, window_length):
    # this method start to run ensemble strategy
    return

import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from config import config
from env.environment_trade import StockEnvTrade
from utility import preprocessor, run_model
from utility.train import train_A2C, train_DDPG, train_PPO
from utility.validate import validate


def ensemble_strategy(model, data, window_length):
    # this method start to run ensemble strategy
    model_used = [model]
    a2c_sharpe_list = []
    ppo_shapre_list = []
    ddpg_sharpe_list = []
    balance_list = [config.INITIAL_AMOUNT]
    shares_list = [[[0] * 10]]

    trade_date = data.Index.unique()

    for i in range(window_length, len(trade_date), window_length):
        start_idx = (i - window_length) * 10
        end_idx = i * 10

        # to do list: turbulence settings
        historical_turbulence = data.iloc[start_idx:end_idx, :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset="Index")
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        # start ensemble strategy
        window_data = preprocessor.data_split(df=data, start=i - window_length, end=i)
        train_env = DummyVecEnv(
            [
                lambda: StockEnvTrade(
                    df=data, initial_amount=balance_list[-1], shares=shares_list[-1]
                )
            ]
        )
        # start a2c training
        model_a2c = train_A2C(
            env_train=train_env, model_name="a2c_ensemble{}".format(i)
        )
        a2c_reward = validate(model, window_data)
        # trade based on previous model

    return


data_folder_path = config.DATA_FOLDER
train_data, test_data = run_model.data_load(folder_name=data_folder_path)
print("Data loading is complete.")

# model = train_model(train_data)
# print("Data training is complete")

model = A2C.load("trained_models/a2c_train.zip")

ensemble_strategy(model=model, data=test_data, window_length=config.WINDOW_LENGTH)

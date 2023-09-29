import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from utility.env.environment_trade import StockEnvTrade
from utility.env.environment_train import StockEnvTrain


def validate(model, data: pd.DataFrame, balance: float, shares: float) -> float:
    """
    the function validate the trained model in a given period
    :return: sharpe ratio
    """
    env = DummyVecEnv(
        [lambda: StockEnvTrade(df=data, initial_amount=balance, shares=shares)]
    )
    obj = env.reset()
    reward = 0
    for i in range(data["Index"].nunique() - 1):
        action, _states = model.predict(obj)
        obj, reward, done, info = env.step(action)
    return reward[0]

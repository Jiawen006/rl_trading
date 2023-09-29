import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from typing import Tuple

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from env.environment_trade import StockEnvTrade
from utility import preprocessor


def trade(model, data, balance, shares) -> Tuple(float, float, float):
    env = DummyVecEnv(
        [
            lambda: StockEnvTrade(
                df=data, initial_amount=balance, shares=shares, save_file=True
            )
        ]
    )
    obs_trade = env.reset()
    trade_num = data.Index.unique()
    trade_num = len(trade_num)
    reward_list = []
    balance_list = []
    share_list = []
    for i in range(trade_num - 1):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env.step(action)
        balance = balance_list.append(obs_trade[0][0])
        shares = share_list.append(obs_trade[0][41:51])
        reward_list.append(rewards[0])
    return float(balance_list[-1]), list(share_list[-1]), reward_list[-1]

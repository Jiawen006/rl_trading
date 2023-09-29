import os
import sys
from typing import Type, Union

import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utility import preprocessor
from utility.env.environment_trade import StockEnvTrade


def trade(
    model: Union[Type[A2C], Type[PPO], Type[DDPG]],
    data: pd.DataFrame,
    balance: float,
    shares: float,
) -> tuple[float, float, float]:
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

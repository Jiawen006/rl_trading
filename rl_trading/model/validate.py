"""import necessary packages for validation"""
from typing import Type, Union

import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_trading.env.environment_trade import StockEnvTrade


def validation(
    model: Union[Type[A2C], Type[PPO], Type[DDPG]],
    data: pd.DataFrame,
    balance: float,
    shares: float,
) -> float:
    """
    the function validate the trained model in a given period
    :return: sharpe ratio
    """
    env = DummyVecEnv(
        [lambda: StockEnvTrade(_df=data, initial_amount=balance, shares=shares)]
    )
    obj = env.reset()
    reward = 0
    for _ in range(data["Index"].nunique() - 1):
        action, _ = model.predict(obj)
        obj, reward, _, _ = env.step(action)
    return reward[0]

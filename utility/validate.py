from typing import Type, Union

import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utility.env.environment_trade import StockEnvTrade


def validate(
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
        [lambda: StockEnvTrade(df=data, initial_amount=balance, shares=shares)]
    )
    obj = env.reset()
    reward = 0
    for i in range(data["Index"].nunique() - 1):
        action, _states = model.predict(obj)
        obj, reward, done, info = env.step(action)
    return reward[0]

"""import required packages for typing, trading environment and predicting on RL model"""
from typing import List, Type, Union

import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utility.env.environment_trade import StockEnvTrade


def trade(
    model: Union[Type[A2C], Type[PPO], Type[DDPG]],
    data: pd.DataFrame,
    balance: float,
    shares: List[float],
) -> tuple[float, float, float]:
    """
    Trade based on pretrained model

    [input]
    * model     : stable baseline model for trading
    * data      : dataframe with a specific trading period
    * balance   : balance before the trading
    * shares    : shares before the trading

    [output]
    * balance   : Remaining balance after this round of trading.
    * shares    : Number of shares remaining after this round.
    * rewards   : Shape ratio after the trading
    """
    env = DummyVecEnv(
        [
            lambda: StockEnvTrade(
                _df=data, initial_amount=balance, shares=shares, save_file=True
            )
        ]
    )
    obs_trade = env.reset()
    trade_num = data.Index.unique()
    trade_num = len(trade_num)
    reward_list = []
    balance_list = []
    share_list = []
    for _ in range(trade_num - 1):
        action, _ = model.predict(obs_trade)
        obs_trade, rewards, _, _ = env.step(action)
        balance_list.append(obs_trade[0][0])
        share_list.append(obs_trade[0][41:51])
        reward_list.append(rewards[0])
    return float(balance_list[-1]), list(share_list[-1]), reward_list[-1]

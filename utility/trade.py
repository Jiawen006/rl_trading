import os
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import pandas as pd
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from env.environment_trade import StockEnvTrade
from utility import preprocessor


def trade(model, data, balance, shares):
    env = DummyVecEnv(
        [lambda: StockEnvTrade(df=data, initial_amount=balance, shares=shares)]
    )
    obs_trade = env.reset()
    trade_num = data.Index.unique()
    trade_num = len(trade_num)
    reward_list = []
    balance_list = []
    share_list = []
    for i in range(trade_num):
        action, _states = model.predict(obs_trade)
        obs_trade, rewards, dones, info = env.step(action)
        # if i == (trade_num - 2):
        #     print(env_test.render())
        #     last_state = env.render()
        #     print(last_state)
        balance = balance_list.append(obs_trade[0][0])
        shares = share_list.append(obs_trade[0][41:51])
        reward_list.append(rewards[0])
    return float(balance_list[-1]), list(share_list[-1]), reward_list[-1]


# model = "trained_models/2023-09-25 16:41:42.793445/a2c_train.zip"
# model = A2C.load(model)
# data = pd.read_csv("DATA/test_processed.csv")
# # trading_day = data['Index'].nunique()
# # 2200 trading days
# trade_date = data.Index.unique()
# window_length = 90

# balance = 1000000
# shares = [0] * 10

# for i in range(window_length, len(trade_date), window_length):
#     start_idx = (i - window_length) * 10
#     end_idx = i * 10

#     start = i - window_length
#     end = i

#     # trade based on previous model
#     window_data = preprocessor.data_split(df=data, start=i - window_length, end=i)
#     balance, shares = trade(
#         model=model, data=window_data, balance=balance, shares=shares
#     )


# print(balance)
# print(shares)

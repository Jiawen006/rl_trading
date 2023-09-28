import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from config import config
from env.environment_trade import StockEnvTrade
from utility import preprocessor, run_model
from utility.trade import trade
from utility.train import train_A2C, train_DDPG, train_PPO
from utility.validate import validate


def ensemble_strategy(model, data, window_length):
    # this method start to run ensemble strategy
    trade_model = model
    model_history_name = ["A2C"]
    a2c_sharpe_list = []
    ppo_sharpe_list = []
    ddpg_sharpe_list = []
    balance_list = [config.INITIAL_AMOUNT]
    shares_list = [[0] * 10]
    trade_sharpe_list = []

    trade_date = data.Index.unique()

    for i in range(window_length, len(trade_date), window_length):
        start_idx = (i - window_length) * 10
        end_idx = i * 10
        window_data = preprocessor.data_split(df=data, start=i - window_length, end=i)
        train_balance = balance_list[-1]
        train_share = shares_list[-1]

        # to do list: turbulence settings
        historical_turbulence = data.iloc[start_idx:end_idx, :]
        historical_turbulence = historical_turbulence.drop_duplicates(subset="Index")
        historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

        # trade to get current balance and shares holding position
        balance_trade, shares_trade, sharpe_trade = trade(
            model=trade_model,
            data=window_data,
            balance=balance_list[-1],
            shares=shares_list[-1],
        )
        ###########################

        # start ensemble strategy
        train_env = DummyVecEnv(
            [
                lambda: StockEnvTrade(
                    df=data, initial_amount=balance_list[-1], shares=shares_list[-1]
                )
            ]
        )
        model_a2c = train_A2C(
            env_train=train_env, model_name="a2c_ensemble{}".format(i)
        )
        a2c_reward = validate(
            model_a2c, window_data, balance=train_balance, shares=train_share
        )
        a2c_sharpe_list.append(a2c_reward)

        # start ppo training
        model_ppo = train_PPO(
            env_train=train_env, model_name="ppo_ensemble{}".format(i)
        )
        ppo_reward = validate(
            model_ppo, window_data, balance=train_balance, shares=train_share
        )
        ppo_sharpe_list.append(ppo_reward)

        # start ddpg training

        model_ddpg = train_DDPG(
            env_train=train_env, model_name="ddpg_ensemble{}".format(i)
        )
        ddpg_reward = validate(
            model_ddpg, window_data, balance=train_balance, shares=train_share
        )
        ddpg_sharpe_list.append(ddpg_reward)

        # compare to get the best performance
        if a2c_reward > max(ddpg_reward, ppo_reward):
            # a2c win in this round
            trade_model = model_a2c
            model_history_name.append("A2C")
        else:
            if ddpg_reward > ppo_reward:
                # ddpg win in this round
                trade_model = model_ddpg
                model_history_name.append("DDPG")
            else:
                # ppo win in this round
                trade_model = model_ppo
                model_history_name.append("PPO")

        balance_list.append(balance_trade)
        shares_list.append(shares_trade)
        trade_sharpe_list.append(sharpe_trade)

    final_sharpe = trade_sharpe_list[-1]

    return final_sharpe

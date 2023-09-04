"""Module providing a function to calculate the training time."""
import time
from typing import Type, Union

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from rl_trading.configs import config
from rl_trading.env.environment_train import StockEnvTrain
from rl_trading.model import validate


def train_model(data: pd.DataFrame) -> Union[Type[A2C], Type[PPO], Type[DDPG]]:
    """
    Training a robust model before trading

    [input]
    * data      : dataframe that consist of open, high, low, close prices

    [output]
    * model: the model with best sharpe ratio between A2C, DDPG, and PPO
    """
    env_train = DummyVecEnv([lambda: StockEnvTrain(data)])

    validate_data = data.tail(90 * 10)
    validate_data.index = validate_data.Index.factorize()[0]

    model_a2c = a2c_training(
        env_train=env_train,
        model_name="a2c_train",
        model=None,
        pretrain=True,
    )
    a2c_reward = validate.validation(
        model_a2c, validate_data, balance=config.INITIAL_AMOUNT, shares=[0] * 10
    )

    model_ppo = ppo_training(
        env_train=env_train,
        model_name="ppo_train",
        model=None,
        pretrain=True,
    )
    ppo_reward = validate.validation(
        model_ppo, validate_data, balance=config.INITIAL_AMOUNT, shares=[0] * 10
    )

    model_ddpg = ddpg_training(
        env_train=env_train,
        model_name="ddpg_train",
        model=None,
        pretrain=True,
    )
    ddpg_reward = validate.validation(
        model_ddpg, validate_data, balance=config.INITIAL_AMOUNT, shares=[0] * 10
    )

    # validate which model perform best
    if a2c_reward > max(ppo_reward, ddpg_reward):
        return model_a2c
    if ppo_reward > ddpg_reward:
        return model_ppo
    return model_ddpg


def a2c_training(
    env_train,
    model_name: pd.DataFrame,
    model=None,
    pretrain=None,
) -> Type[A2C]:
    """
     Train a A2C model

     [input]
    * env_train      : training environment
    * model_name     : name of the saved files

     [output]
    * model          : A2C model
    """

    start = time.time()
    if model is None:
        model = A2C("MlpPolicy", env_train, **config.A2C_PARAMS)
    if pretrain is True:
        model.learn(total_timesteps=config.A2CTIMESTEPS)
    else:
        model.learn(total_timesteps=config.A2CTIMESTEPS / 10)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (A2C): ", (end - start) / 60, " minutes")
    return model


def ppo_training(
    env_train, model_name: pd.DataFrame, model=None, pretrain=None
) -> Type[PPO]:
    """
     Train a ppo model

     [input]
    * env_train      : training environment
    * model_name     : name of the saved files

     [output]
    * model          : PPO model
    """

    start = time.time()
    if model is None:
        model = PPO("MlpPolicy", env_train, **config.PPO_PARAMS)
    # model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)
    if pretrain is True:
        model.learn(total_timesteps=config.PPOTIMESTEPS)
    else:
        model.learn(total_timesteps=config.PPOTIMESTEPS / 10)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO): ", (end - start) / 60, " minutes")
    return model


def ddpg_training(
    env_train, model_name: pd.DataFrame, model=None, pretrain=None
) -> Type[DDPG]:
    """
     Train a ddpg model

     [input]
    * env_train      : training environment
    * model_name     : name of the saved files

     [output]
    * model          : DDPG model
    """

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
    )

    start = time.time()
    if model is None:
        model = DDPG(
            "MlpPolicy", env_train, action_noise=action_noise, **config.DDPG_PARAMS
        )
    if pretrain is True:
        model.learn(total_timesteps=config.DDPG_TIMESTEPS)
    else:
        model.learn(total_timesteps=config.DDPG_TIMESTEPS / 10)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (DDPG): ", (end - start) / 60, " minutes")
    return model

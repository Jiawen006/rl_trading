import time
from typing import TypeVar

import numpy as np
import pandas as pd
from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

from config import config
from utility.env.environment_train import StockEnvTrain
from utility.validate import validate


def run_train(data: pd.DataFrame) -> TypeVar:
    env_train = DummyVecEnv([lambda: StockEnvTrain(data)])

    data = pd.read_csv("DATA/train_processed.csv")
    validate_data = data.tail(90 * 10)
    validate_data.index = validate_data.Index.factorize()[0]

    model_a2c = train_A2C(env_train=env_train, model_name="a2c_train")
    a2c_reward = validate(
        model_a2c, validate_data, balance=config.INITIAL_AMOUNT, shares=[0] * 10
    )

    model_ppo = train_PPO(env_train=env_train, model_name="ppo_train")
    ppo_reward = validate(
        model_ppo, validate_data, balance=config.INITIAL_AMOUNT, shares=[0] * 10
    )

    model_ddpg = train_DDPG(env_train=env_train, model_name="ddpg_train")
    ddpg_reward = validate(
        model_ddpg, validate_data, balance=config.INITIAL_AMOUNT, shares=[0] * 10
    )

    # validate which model perform best
    if a2c_reward > max(ppo_reward, ddpg_reward):
        return model_a2c
    else:
        if ppo_reward > ddpg_reward:
            return model_ppo
        else:
            return model_ddpg


def train_A2C(env_train, model_name: pd.DataFrame) -> TypeVar:
    """A2C model"""

    start = time.time()
    model = A2C("MlpPolicy", env_train, **config.A2C_PARAMS)
    model.learn(total_timesteps=config.a2ctimesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (A2C): ", (end - start) / 60, " minutes")
    return model


def train_PPO(env_train, model_name: pd.DataFrame) -> TypeVar:
    """PPO model"""

    start = time.time()
    model = PPO("MlpPolicy", env_train, **config.PPO_PARAMS)
    # model = PPO2('MlpPolicy', env_train, ent_coef = 0.005)

    model.learn(total_timesteps=config.ppotimesteps)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (PPO): ", (end - start) / 60, " minutes")
    return model


def train_DDPG(env_train, model_name: pd.DataFrame) -> TypeVar:
    """DDPG model"""

    # add the noise objects for DDPG
    n_actions = env_train.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(
        mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions)
    )

    start = time.time()
    model = DDPG(
        "MlpPolicy", env_train, action_noise=action_noise, **config.DDPG_PARAMS
    )

    model.learn(total_timesteps=config.ddpgtimestep)
    end = time.time()

    model.save(f"{config.TRAINED_MODEL_DIR}/{model_name}")
    print("Training time (DDPG): ", (end - start) / 60, " minutes")
    return model

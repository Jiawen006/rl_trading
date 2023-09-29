import os
from typing import Tuple, Type, Union

import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Index, Series
from stable_baselines3 import A2C, DDPG, PPO

from utility import config, preprocessor
from utility.ensemble import ensemble_strategy
from utility.train import run_train


def run() -> None:
    """
    The baseline of the whole program

    [input]
    * None

    [output]
    * None
    """
    data_folder_path = config.DATA_FOLDER
    train_data, test_data = data_load(folder_name=data_folder_path)
    print("Data loading is complete.")

    model = train_model(train_data)
    print("Data training is complete")

    final_sharpe = ensemble_strategy(
        model=model, data=test_data, window_length=config.WINDOW_LENGTH
    )

    print("Trading ends. Final Sharpe Ratio is {}".format(final_sharpe))


def data_load(folder_name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset from the system and preprocess the data

    [input]
    * folder name: string

    [output]
    * train_data: pandas.DataFrame (including technical indicator and turbulence)
    * test_data: pandas.DataFrame (including technical indicator and turbulence)
    """
    # check whether folder exist
    if not os.path.exists(folder_name):
        raise Exception("DATA Folder not found.")
    # check whether processed data exist
    if os.path.exists(config.TRAINING_DATA_NAME):
        train_data = pd.read_csv(config.TRAINING_DATA_NAME, index_col=0)
    else:
        train_data = preprocessor.preprocess_pipeline(config.TRAINING_DATA_FOLDER)
        train_data.to_csv(config.TRAINING_DATA_NAME, index=True)

    if os.path.exists(config.TEST_DATA_NAME):
        test_data = pd.read_csv(config.TEST_DATA_NAME, index_col=0)
    else:
        test_data = preprocessor.preprocess_pipeline(config.TEST_DATA_FOLDER)
        test_data.to_csv(config.TEST_DATA_NAME, index=True)

    return train_data, test_data


def train_model(data: pd.DataFrame) -> Union[Type[A2C], Type[PPO], Type[DDPG]]:
    """
    Train a model before trading

    [input]
    * Data: pandas.Dataframe (Training Data)

    [output]
    * model: Stable Baseline 3 Model
    """

    model = run_train(data)
    return model
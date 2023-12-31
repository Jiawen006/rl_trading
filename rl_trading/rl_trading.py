"""import os package for file location retrieval and other local packages for baseline execution"""
import os
from typing import Tuple

import pandas as pd

from rl_trading.configs import config
from rl_trading.model import ensemble, train
from rl_trading.utils import preprocessor


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

    model = train.train_model(train_data)
    print("Data training is complete")

    final_sharpe = ensemble.ensemble_strategy(
        model=model, data=test_data, window_length=config.WINDOW_LENGTH
    )

    print(f"Trading ends. Final Sharpe Ratio is {final_sharpe}")


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
        raise FileNotFoundError("DATA Folder not found.")
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

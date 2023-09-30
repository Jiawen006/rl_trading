import os

import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from utility import config


def data_split(_df: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """
    split the dataset for trading in a sub-window

    [input]
    * _df       : dataframes with all testing data
    * start     : window start date
    * end       : window end date

    [output]
    * data      : sub dataframe
    """

    data = _df.iloc[start * 10 : end * 10, :]
    data = data.sort_values(["Index", "Series"], ignore_index=True)
    data.index = data.Index.factorize()[0]

    return data


def preprocess_pipeline(folder_path: str) -> pd.DataFrame:
    """
    general pipeline of precess data
    step 1: load dataset
    step 2: process all files into one dataframe
    step 3: add technical indicators
    step 4: add turbulence to the dataset

    [input]
    * folder_path   : folder name

    [output]
    * dataframe     : processed dataframe consist of all datasets
    """
    result = pd.DataFrame()
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            single_df = single_preprocess(file_path)
            result = pd.concat([result, single_df]).reset_index(drop=True)
    for col in result.columns:
        result[col] = result[col].round(6)
    result = result.sort_values(["Index", "Series"]).reset_index(drop=True)

    # add turbulence
    result = add_turbulence(result)
    result = result.sort_values(["Index", "Series"], ignore_index=True)
    # data  = data[final_columns]
    result.index = result.Index.factorize()[0]
    return result


def single_preprocess(file_name: str) -> pd.DataFrame:
    """data preprocessing pipeline"""

    _df = load_dataset(file_name=file_name)
    # add technical indicators using stockstats
    _df = add_technical_indicator(_df)
    # fill the missing values at the beginning
    _df.bfill(inplace=True)
    return _df


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load dataset from the directory

    [input]
    * file name     : file name

    [output]
    * dataframe     : loaded dataframe
    """
    _data = pd.read_csv(file_name)

    # convert the Index column into integer
    _data["Index"] = _data["Index"].str.replace("-", "").astype(int)

    series_num = int(file_name.split("/")[-1].replace(".csv", ""))
    _data = _data.assign(Series=series_num)
    return _data


def add_technical_indicator(_df: pd.DataFrame) -> pd.DataFrame:
    """
    calcualte technical indicators
    use stockstats package to add technical indicators

    [input]
    * _df            : dataframe

    [output]
    * _df           : dataframe with technical indicator
    """

    stock = Sdf.retype(_df.copy())

    _macd = pd.DataFrame()
    _rsi = pd.DataFrame()
    _cci = pd.DataFrame()
    _dx = pd.DataFrame()

    ## macd
    temp_macd = stock["macd"]
    temp_macd = pd.DataFrame(temp_macd)
    _macd = pd.concat([temp_macd, _macd]).reset_index(drop=True)
    ## rsi
    temp_rsi = stock["rsi_30"]
    temp_rsi = pd.DataFrame(temp_rsi)
    _rsi = pd.concat([temp_rsi, _rsi]).reset_index(drop=True)
    ## cci
    temp_cci = stock["cci_30"]
    temp_cci = pd.DataFrame(temp_cci)
    _cci = pd.concat([temp_cci, _cci]).reset_index(drop=True)
    ## adx
    temp_dx = stock["dx_30"]
    temp_dx = pd.DataFrame(temp_dx)
    _dx = pd.concat([temp_dx, _dx]).reset_index(drop=True)

    _df["macd"] = _macd
    _df["rsi"] = _rsi
    _df["cci"] = _cci
    _df["adx"] = _dx

    return _df


def add_turbulence(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add turbulence index from a precalcualted dataframe

    [input]
    * _df            : dataframe

    [output]
    * _df           : dataframe with turbulence
    """
    turbulence_index = calcualte_turbulence(_df)
    _df = _df.merge(turbulence_index, on="Index")
    _df = _df.sort_values(["Index", "Series"]).reset_index(drop=True)
    return _df


def calcualte_turbulence(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add turbulence index from a precalcualted dataframe

    [input]
    * _df               : dataframe

    [output]
    * turbulence_index  : calculate turbulence for each day
    """

    df_price_pivot = _df.pivot(index="Index", columns="Series", values="Close")
    unique_date = _df.Index.unique()
    # start after a period
    start = config.TURBULENCE_START
    turbulence_index = [0] * start
    count = 0
    for i in range(start, len(unique_date)):
        current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
        hist_price = df_price_pivot[
            [n in unique_date[0:i] for n in df_price_pivot.index]
        ]
        cov_temp = hist_price.cov()
        current_temp = current_price - np.mean(hist_price, axis=0)
        temp = current_temp.values.dot(np.linalg.inv(cov_temp)).dot(
            current_temp.values.T
        )
        if temp > 0:
            count += 1
            if count > 2:
                turbulence_temp = temp[0][0]
            else:
                # avoid large outlier because of the calculation just begins
                turbulence_temp = 0
        else:
            turbulence_temp = 0
        turbulence_index.append(turbulence_temp)

    turbulence_index = pd.DataFrame(
        {"Index": df_price_pivot.index, "turbulence": turbulence_index}
    )
    turbulence_index["turbulence"] = turbulence_index["turbulence"].round(6)
    return turbulence_index

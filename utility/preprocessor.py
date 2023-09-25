import os

import config
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df.iloc[start * 10, end * 10]
    data = data.sort_values(["Index", "Series"], ignore_index=True)
    data.index = data.Index.factorize()[0]

    return data


def preprocess_pipeline(folder_path):
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


def single_preprocess(file_name):
    """data preprocessing pipeline"""

    df = load_dataset(file_name=file_name)
    # add technical indicators using stockstats
    df = add_technical_indicator(df)
    # fill the missing values at the beginning
    df.bfill(inplace=True)
    return df


def load_dataset(*, file_name: str) -> pd.DataFrame:
    """
    load csv dataset from path
    :return: (df) pandas dataframe
    """
    _data = pd.read_csv(file_name)

    # convert the Index column into integer
    _data["Index"] = _data["Index"].str.replace("-", "").astype(int)

    series_num = int(file_name.split("/")[-1].replace(".csv", ""))
    _data = _data.assign(Series=series_num)
    return _data


def add_technical_indicator(df):
    """
    calcualte technical indicators
    use stockstats package to add technical indicators
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """

    stock = Sdf.retype(df.copy())

    macd = pd.DataFrame()
    rsi = pd.DataFrame()
    cci = pd.DataFrame()
    dx = pd.DataFrame()

    ## macd
    temp_macd = stock["macd"]
    temp_macd = pd.DataFrame(temp_macd)
    macd = pd.concat([temp_macd, macd]).reset_index(drop=True)
    ## rsi
    temp_rsi = stock["rsi_30"]
    temp_rsi = pd.DataFrame(temp_rsi)
    rsi = pd.concat([temp_rsi, rsi]).reset_index(drop=True)
    ## cci
    temp_cci = stock["cci_30"]
    temp_cci = pd.DataFrame(temp_cci)
    cci = pd.concat([temp_cci, cci]).reset_index(drop=True)
    ## adx
    temp_dx = stock["dx_30"]
    temp_dx = pd.DataFrame(temp_dx)
    dx = pd.concat([temp_dx, dx]).reset_index(drop=True)

    df["macd"] = macd
    df["rsi"] = rsi
    df["cci"] = cci
    df["adx"] = dx

    return df


def add_turbulence(df):
    """
    add turbulence index from a precalcualted dataframe
    :param data: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    turbulence_index = calcualte_turbulence(df)
    df = df.merge(turbulence_index, on="Index")
    df = df.sort_values(["Index", "Series"]).reset_index(drop=True)
    return df


def calcualte_turbulence(df):
    # can add other market assets

    df_price_pivot = df.pivot(index="Index", columns="Series", values="Close")
    unique_date = df.Index.unique()
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

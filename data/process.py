import os

import numpy as np
import pandas as pd
from pytorch_forecasting.data.examples import get_stallion_data

filename = "sensorData.csv"
dirname = os.path.dirname(os.path.abspath(__file__))
filename_with_path = os.path.join(dirname, filename)

dtype = {"result": str, "table": str, "_time": str, "moisture": "float64"}


def read_file():
    return pd.read_csv(
        filename_with_path,
        sep=",",
        encoding="utf-8",
        dtype=dtype,
        parse_dates=["_time"],
    )


def get_influx_dataset(values=[], fill_missing=True):
    if not values:
        file_dataset = read_file()
        values = file_dataset["moisture"]
    dataframe = pd.DataFrame(
        dict(
            value=values,  # Extracting the values from the '_value' column
            group=0,  # Setting the 'group' column to 0
            time_idx=range(len(values)),  # Setting 'time_idx' to an increasing integer
        )
    )
    # dataframe.set_index('time_idx', inplace=True)
    if fill_missing:
        dataframe = linear_fill_missing(dataframe)
    return dataframe


# Fills mising values by linear interpolation
# Drops remaining missing values (missing values at dataset start can't be interpolated)
def linear_fill_missing(dataset):
    dataset["value"] = dataset["value"].interpolate(method="linear")
    return dataset.dropna()


def get_sawtooth_dataset(amount_interval, length_interval=10):
    max_range = amount_interval * length_interval
    sawtooth_values = [(i % length_interval) for i in range(1, max_range)]
    return get_influx_dataset(sawtooth_values)


def get_linear_dataset(length):
    linear_values = range(length)
    return get_influx_dataset(linear_values)


# Dataset used in the pytorch_forecasting tutorial
def get_stallion_dataset() -> pd.DataFrame:
    dataset = get_stallion_data()
    # add time index
    dataset["time_idx"] = dataset["date"].dt.year * 12 + dataset["date"].dt.month
    dataset["time_idx"] -= dataset["time_idx"].min()

    # add additional features
    dataset["month"] = dataset.date.dt.month.astype(str).astype(
        "category"
    )  # categories have be strings
    dataset["log_volume"] = np.log(dataset.volume + 1e-8)
    dataset["avg_volume_by_sku"] = dataset.groupby(
        ["time_idx", "sku"], observed=True
    ).volume.transform("mean")
    dataset["avg_volume_by_agency"] = dataset.groupby(
        ["time_idx", "agency"], observed=True
    ).volume.transform("mean")

    # we want to encode special days as one variable and thus need to first reverse one-hot encoding
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    dataset[special_days] = (
        dataset[special_days]
        .apply(lambda x: x.map({0: "-", 1: x.name}))
        .astype("category")
    )
    dataset.sample(10, random_state=521)
    return dataset

import numpy as np
from pandas import DataFrame, Series
from pytorch_forecasting.data.examples import get_stallion_data

from data.analyse import calculate_season_length
from data.from_db import read_file


def get_influx_dataset(
    resolution: str, should_fill_missing: bool = True, should_normalize: bool = False
) -> DataFrame:
    """
    Retrieves a dataset from InfluxDB with the specified resolution.
    Parameters:
        resolution (str): The resolution of the dataset., p.ex 30m, 6h, 1d
        fill_missing (bool, optional): Flag indicating whether to fill missing values in the dataset. Defaults to True.
    Returns:
        pandas.DataFrame: The retrieved dataset.
    """

    file_dataset = read_file(resolution)
    values = file_dataset["moisture"]
    if should_normalize:
        values = _normalize_dataset(values)
    dataframe = DataFrame(
        dict(
            value=values,
            group=0,
            time_idx=range(len(values)),  # TODO use actual dates instead of range
        )
    )
    if should_fill_missing:
        dataframe["value"] = _linear_fill_missing(dataframe["value"])
    return dataframe


def get_sawtooth_dataset(amount_interval: int, length_interval: int = 10) -> DataFrame:
    """
    Generates a dataset conatining a sawtooth function.
    Parameters:
    - amount_interval (int): The number of intervals to generate.
    - length_interval (int): The length of each interval. Default is 10.
    Returns:
    - DataFrame: The generated sawtooth dataset.
    """
    max_range = amount_interval * length_interval
    sawtooth_values = [(i % length_interval) for i in range(1, max_range)]
    return DataFrame(
        dict(value=sawtooth_values, group=0, time_idx=range(len(sawtooth_values)))
    )


def get_stallion_dataset() -> DataFrame:
    """
    Generates the dataset used in the pytorch_forecasting tutorial
    Returns:
        DataFrame: Dataset used in the pytorch_forecasting tutorial
    """
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


# TODO implement
def get_influx_chained_seasons_dataset(resolution: str) -> DataFrame:
    file_dataset = read_file(resolution)
    season_length = calculate_season_length(file_dataset)


def _normalize_dataset(dataseries: Series) -> Series:
    """
    Normalizes a data series via min-max scaling.
    Parameters:
        dataseries (pandas.Series): The dataset to be normalized.
    Returns:
        pandas.Series: The normalized dataseries.
    """
    return (dataseries - dataseries.min()) / (dataseries.max() - dataseries.min())


def _linear_fill_missing(
    dataset: Series,
) -> Series:
    """
    Drops beginning and fills missing values in a dataset.
    Parameters:
        dataset (pandas.Series): The dataset to be processed.
    Returns:
        pandas.DataFrame: The dataset with missing values filled using linear interpolation.
    """
    dataset = dataset.interpolate(method="linear")
    return dataset.dropna()

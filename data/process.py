import numpy as np
from pandas import DataFrame, Series
from pytorch_forecasting.data.examples import get_stallion_data

from data.from_db import read_file


def get_influx_dataset(
    resolution: str,
    should_fill_missing: bool = True,
    should_normalize: bool = True,
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


def get_sawtooth_dataset(
    amount_intervals: int,
    steps_per_interval: int = 10,
    interval_length: int = 10,
) -> DataFrame:
    """
    Generate a sawtooth dataset in range [1, interval_length].
    Parameters:
        amount_intervals (int): Number of intervals to generate.
        steps_per_interval (int, optional): Number of data points per interval. Default is 10.
        interval_length (int, optional): Length of each interval. Default is 10.
    Returns:
        DataFrame: A DataFrame containing the sawtooth values, group, and time index.
    """
    increment = interval_length / steps_per_interval
    sawtooth_values = Series(
        np.tile(np.arange(1, interval_length, increment), amount_intervals)
    )
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


def _normalize_dataset(dataseries: Series) -> Series:
    """
    Normalizes a data series via min-max scaling to a range of 1 to 10.

    Parameters:
    dataseries (pandas.Series): The dataset to be normalized.

    Returns:
    pandas.Series: The normalized dataseries with values between 1 and 10.
    """
    min_val = dataseries.min()
    max_val = dataseries.max()

    return 1 + 9 * (dataseries - min_val) / (max_val - min_val)


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

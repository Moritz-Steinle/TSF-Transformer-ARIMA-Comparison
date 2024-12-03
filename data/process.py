import numpy as np
from pandas import DataFrame, Series
from pytorch_forecasting.data.examples import get_stallion_data

from data.from_db import read_file


def get_csv_dataset(
    filename: str,
    input_values_column_name: str = "moisture",
    should_fill_missing: bool = True,
    should_normalize: bool = True,
) -> DataFrame:
    """
    Reads a dataset from a csv file, processes it, and returns it as a DataFrame.
    Parameters:
        filename (str): The path to the file containing the dataset.
        input_values_column_name (str, optional): The name of the column containing the input values.
            Default is 'moisture'.
        should_fill_missing (bool, optional): If missing values in the dataset should be filled using
            linear interpolation. Default is True.
        should_normalize (bool, optional): If should be normalized. Default is True.
    Returns:
        DataFrame: A DataFrame containing the processed dataset with columns 'value', 'group', and 'time_idx'.
            These colums are required by pytorch-forecasting.
    """
    file_dataset = read_file(filename)
    values = file_dataset[input_values_column_name]
    if should_fill_missing:
        values = _linear_fill_missing(values)
    if should_normalize:
        values = _normalize_dataset(values)
    return DataFrame(
        dict(
            value=values,
            group=0,
            time_idx=range(len(values)),
        )
    )


def get_sawtooth_dataset(
    amount_intervals: int,
    steps_per_interval: int = 10,
    max_value: int = 10,
    should_normalize: bool = True,
) -> DataFrame:
    """
    Generate a sawtooth dataset in range [1, interval_length].
    Parameters:
        amount_intervals (int): Number of intervals to generate.
        steps_per_interval (int, optional): Number of data points per interval. Default is 10.
        max_value (int, optional): Maximum value (inclusive). Default is 10.
        should_normalize (bool, optional): If should be normalized. Default is True.
    Returns:
        DataFrame: A DataFrame containing the sawtooth values, group, and time index.
    """
    points = np.linspace(1, max_value, steps_per_interval)
    sawtooth_values = Series(np.tile(points, amount_intervals))
    if should_normalize:
        sawtooth_values = _normalize_dataset(sawtooth_values)

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


def _normalize_dataset(series: Series, target_min: int = 1, target_max=10) -> Series:
    """
    Normalizes a data series via min-max scaling to a range of 1 to 10.
    Parameters:
        series (pandas.Series): The dataset to be normalized.
        target_min (int, optional): The minimum value of the target range. Default is 1.
        target_max (int, optional): The maximum value of the target range. Default is 10.
    Returns:
        pandas.Series: The normalized dataseries with values between 1 and 10.
    """
    min_val = series.min()
    max_val = series.max()

    target_max = target_max - target_min
    return target_min + target_max * (series - min_val) / (max_val - min_val)


def _linear_fill_missing(series: Series) -> Series:
    """
    Fills missing values in a series using linear interpolation.
    Beginning and ending missing values are dropped as they cannot be interpolated.
    Parameters:
        series (pandas.Series): The dataset to be processed.
    Returns:
        pandas.DataFrame: The dataset with missing values filled using linear interpolation.
    """
    series = series.interpolate(method="linear")
    return series.dropna()

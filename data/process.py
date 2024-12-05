import os

import numpy as np
from pandas import DataFrame, Series, read_csv

from config import config


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
    file_dataset = _csv_to_dataframe(filename)
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


def get_synthetic_dataset(
    amount_intervals: int,
    steps_per_interval: int = 10,
    max_value: int = 10,
    should_normalize: bool = True,
) -> DataFrame:
    """
    Generate a synthetic dataset with a sawtooth function in range [1, interval_length].
    Parameters:
        amount_intervals (int): Number of intervals to generate.
        steps_per_interval (int, optional): Number of data points per interval. Default is 10.
        max_value (int, optional): Maximum value (inclusive). Default is 10.
        should_normalize (bool, optional): If should be normalized. Default is True.
    Returns:
        DataFrame: The generated dataset.
    """
    points = np.linspace(1, max_value, steps_per_interval)
    sawtooth_values = Series(np.tile(points, amount_intervals))
    if should_normalize:
        sawtooth_values = _normalize_dataset(sawtooth_values)

    return DataFrame(
        dict(value=sawtooth_values, group=0, time_idx=range(len(sawtooth_values)))
    )


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


def _csv_to_dataframe(filename: str) -> DataFrame:
    dirname = os.path.dirname(os.path.abspath(__file__))
    filename_with_path = os.path.join(dirname, config.data_file_path, f"{filename}.csv")
    dtype = {"result": str, "table": str, "_time": str, "moisture": "float64"}
    return read_csv(
        filename_with_path,
        sep=",",
        encoding="utf-8",
        dtype=dtype,
        parse_dates=["_time"],
    )

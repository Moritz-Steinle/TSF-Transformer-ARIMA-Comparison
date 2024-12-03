import matplotlib.pyplot as plt
from pandas import Series
from statsmodels.graphics.tsaplots import acf
from statsmodels.tsa.stattools import adfuller


def analyse_series(raw_series: Series, filled_series: Series = None) -> None:
    """
    Analyzes the given series and prints various statistics.
    Parameters:
        raw_series (DataFrame): The the raw (not filled) series to be analyzed.
        filled_dataset (DataFrame, optional): The dataset after filling missing values. Default is an empty list.
    Prints:
    - The length of the raw dataset.
    - The percentage of missing values in the dataset.
    - The minimum and maximum values in the 'value' column of the dataset.
    - The number of dropped datapoints if a filled dataset is provided
        (missing datapoints at the beginning and end of the dataset cannot be interpolated and are dropped)
    """

    length_dataset = len(raw_series)
    print(f"Raw series length: {length_dataset}")
    if filled_series is not None:
        length_filled_dataset = len(filled_series)
        print(f"Filled series length: {length_filled_dataset}")
        count_dropped_rows = length_dataset - length_filled_dataset
        print(f"Dropped datapoints: {count_dropped_rows}")

    print(
        f"{_get_missing_value_percentage(raw_series)}% of values missing in raw series"
    )
    print(f"Values are in range [{raw_series.min()}, {raw_series.max()}]")


def plot_series(series: Series, tail: int = None) -> None:
    """
    Plots the given series using matplotlib.
    Parameters:
        series (Series): The dataset to be plotted.
        tail (int, optional): If provided, only the last 'tail' number of entries in the dataset will be plotted.
    Returns:
        None
    """
    if tail is not None:
        series = series.tail(tail)
    _, ax = plt.subplots(figsize=(12, 6))
    series.plot(ax=ax, color="blue")
    plt.tight_layout()
    plt.show()


# Adapted from https://machinelearningmastery.com/time-series-data-stationary-python/
def augmented_dickey_fuller_test(series: Series) -> float:
    """
    Perform the Augmented Dickey-Fuller test on a given data series to check for stationarity.
    A p-value of less than 0.05 indicates that the dataset is stationary
    Parameters:
        series (Series): The time series data to be tested.
    Returns:
        float: The p-value from the Augmented Dickey-Fuller test.
    The function prints the ADF Statistic and the p-value.
    """

    result = adfuller(series)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[1]


def calculate_season_length(series: Series) -> int:
    """
    Finds the season length of the dataset by calculating the autocorrelation function (ACF) and finding the first peak.
    Args:
        series (Series): dataset to find the season length of
    Returns:
        int: the season length of the dataset. Returns 1 if no season length is detected, to comply with the ARIMA minimum season length of 1.
    """
    # Calculate the ACF
    lag_acf = acf(series, nlags=len(series) // 2)
    # Find the peaks in the ACF
    from scipy.signal import find_peaks

    peaks, _ = find_peaks(lag_acf)

    if len(peaks) == 0:
        print("No seasonal period detected")
        return 1
    # The first peak after lag 0 is likely to be the seasonal period
    seasonal_period = peaks[peaks > 0][0]
    print(f"Detected seasonal period: {seasonal_period}")
    return seasonal_period


def _get_missing_value_percentage(series: Series) -> float:
    """
    Calculate the percentage of missing values in a pandas Series.
    Args:
        series (Series): The pandas Series for which to calculate the missing value percentage.
    Returns:
        float: The percentage of missing values in the Series, rounded to 8 decimal places.
    """
    length_dataset = len(series)
    count_missing_value = series.isnull().sum()
    exact_percentage = count_missing_value / length_dataset
    return round(exact_percentage, 8) * 100

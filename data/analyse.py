import matplotlib.pyplot as plt
from pandas import Series
from statsmodels.graphics.tsaplots import acf
from statsmodels.tsa.stattools import adfuller


def get_missing_value_percentage(dataset):
    length_dataset = dataset.shape[0]
    count_missing_value = dataset["value"].isnull().sum()
    exact_percentage = count_missing_value / length_dataset
    return round(exact_percentage, 3) * 100


def analyse_dataset(dataset, filled_dataset=[]):
    length_dataset = dataset.shape[0]
    print(f"Raw dataset length: {length_dataset}")
    missing_value_percentage = get_missing_value_percentage(dataset)
    print(f"Percentage of missing values: {missing_value_percentage}")
    print(f"Min value = {dataset['value'].min()}, max value = {dataset['value'].max()}")
    if len(filled_dataset) > 0:
        length_filled_dataset = filled_dataset.shape[0]
        count_dropped_rows = length_dataset - length_filled_dataset
        print(f"Dropped datapoints: {count_dropped_rows}")


def plot_dataset(dataset, key="value"):
    moisture_values = dataset[key].tolist()
    _, axes = plt.subplots()
    axes.plot(moisture_values, color="blue")
    plt.show()


# Adapted from https://machinelearningmastery.com/time-series-data-stationary-python/
def augmented_dickey_fuller_test(dataset: Series) -> float:
    """Runs the Augmented Dickey Fuller test on the dataset

    Args:
        dataset Series: dataset to run the test on

    Returns:
        float: p-value of the test
    """
    result = adfuller(dataset)
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    return result[1]


def calculate_season_length(dataset: Series) -> int:
    """Finds the season length of the dataset

    Args:
        dataset Series: dataset to find the season length of
    """
    # Calculate the ACF
    lag_acf = acf(dataset, nlags=len(dataset) // 2)
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

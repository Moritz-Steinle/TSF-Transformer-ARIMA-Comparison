import matplotlib.pyplot as plt
from pandas import Series
from statsmodels.graphics.tsaplots import acf


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
    if not filled_dataset.empty:
        length_filled_dataset = filled_dataset.shape[0]
        count_dropped_rows = length_dataset - length_filled_dataset
        print(f"Dropped datapoints: {count_dropped_rows}")


def plot_dataset(dataset):
    moisture_values = dataset["value"].tolist()
    _, axes = plt.subplots()
    axes.plot(moisture_values, color="blue")
    plt.show()


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

    # The first peak after lag 0 is likely to be the seasonal period
    seasonal_period = peaks[peaks > 0][0]
    print(f"Detected seasonal period: {seasonal_period}")
    return seasonal_period

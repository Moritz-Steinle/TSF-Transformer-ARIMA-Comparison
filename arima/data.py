from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ARIMA as pmdARIMA
from statsmodels.graphics.tsaplots import acf

from .interface import ArimaDatasets, ArimaOrder


def train_test_split_dataset(
    dataset: Series, max_prediction_length: int
) -> ArimaDatasets:
    """
    Splits the dataset into training and testing sets for ARIMA model.
    Args:
        dataset (Series): The inp²ut dataset.
        max_prediction_length (int): Where to split the input dataset.
    Returns:
        ArimaDatasets: An object containing the training and testing datasets.
    """

    train_dataset = dataset.head(-max_prediction_length)
    test_dataset = dataset.tail(max_prediction_length)
    return ArimaDatasets(train_dataset=train_dataset, test_dataset=test_dataset)


def find_best_order(
    dataset: Series,
    should_calculate_season_length=True,
    season_length=None,
    quiet=True,
) -> ArimaOrder:
    """Runs autoarima that tries out different pdq values and prints the best ones

    Args:
        dataset Series: dataset to run autoarima on
        season_length int: season length for the dataset
    """
    if should_calculate_season_length:
        season_length = calculate_season_length(dataset)
    stepwise_model: pmdARIMA = auto_arima(
        dataset,
        seasonal_test="ch",  # Default OCSB throws ValueError: All lag values up to 'maxlag' produced singular matrices.
        trace=quiet,
        seasonal=True,
        m=season_length,
        suppress_warnings=True,
    )
    if not quiet:
        stepwise_model.summary()
    return ArimaOrder(stepwise_model.order, stepwise_model.seasonal_order)


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

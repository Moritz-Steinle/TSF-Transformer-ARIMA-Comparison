from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ARIMA as pmdARIMA
from threadpoolctl import threadpool_limits

from data.analyse import calculate_season_length

from .interface import ArimaDatasets, ArimaOrder


def train_test_split_dataset(dataset: Series, prediction_length: int) -> ArimaDatasets:
    """
    Splits the dataset into training and testing sets, depending on the prediction length.
    Args:
        dataset (Series): The input dataset.
        prediction_length (int): Where to split the input dataset.
    Returns:
        ArimaDatasets: An object containing the training and testing datasets.
    """
    train_dataset = dataset.head(-prediction_length)
    validation_dataset = dataset.tail(prediction_length)
    if not train_dataset:
        raise ValueError(
            f"Prediction length {prediction_length} is too large for dataset of length {len(dataset)}"
        )
    return ArimaDatasets(
        train_dataset=train_dataset, validation_dataset=validation_dataset
    )


def find_best_order(
    dataset: Series,
    is_data_stationary: bool = False,
    max_p=2,
    max_q=2,
    season_length: int = None,
    quiet: bool = True,
) -> ArimaOrder:
    """
    Find the best ARIMA order for the given dataset using a stepwise search.
    Parameters:
        dataset (Series): The time series dataset for which to find the best ARIMA order.
        is_data_stationary (bool, optional): If True, the d and D parameters will be set to 0 to disable differencing.
            Default is False.
        season_length (int, optional): The length of the seasonal cycle.
            If None, it will be calculated automatically. Default is None.
        quiet (bool, optional): If True, suppresses the output of the stepwise search process.
            Default is True.
    Returns:
        ArimaOrder: The suggested ARIMA order and seasonal order.
    """
    if season_length is None:
        season_length = calculate_season_length(dataset)
    # Limits CPU Usage to prevent the process from being killed
    with threadpool_limits(limits=3, user_api="blas"):
        stepwise_model: pmdARIMA = auto_arima(
            dataset,
            trace=quiet,
            seasonal=True,
            m=season_length,
            suppress_warnings=True,
            max_p=max_p,
            D=0 if is_data_stationary else None,
            d=0 if is_data_stationary else None,
            max_q=max_q,
        )
    if not quiet:
        stepwise_model.summary()
    return ArimaOrder(stepwise_model.order, stepwise_model.seasonal_order)

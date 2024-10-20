from pandas import Series
from pmdarima import auto_arima
from pmdarima.arima import ARIMA as pmdARIMA
from threadpoolctl import threadpool_limits

from data.analyse import calculate_season_length

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
    validation_dataset = dataset.tail(max_prediction_length)
    if len(train_dataset) == 0:
        raise ValueError(
            f"Prediction length {max_prediction_length} is too large for dataset of length {len(dataset)}"
        )
    return ArimaDatasets(
        train_dataset=train_dataset, validation_dataset=validation_dataset
    )


def find_best_order(
    dataset: Series,
    season_length: int = None,
    quiet: bool = True,
) -> ArimaOrder:
    """Runs autoarima that tries out different pdq values and prints the best ones

    Args:
        dataset Series: dataset to run autoarima on
        season_length int: season length for the dataset
    """
    if season_length is None:
        season_length = calculate_season_length(dataset)
    with threadpool_limits(
        limits=3, user_api="blas"
    ):  # Limits CPU Usage to 1 cores to prevent killing the process
        stepwise_model: pmdARIMA = auto_arima(
            dataset,
            trace=quiet,
            seasonal=True,
            m=season_length,
            suppress_warnings=True,
            max_p=2,
            D=0,
            d=0,
            max_q=2,
        )
    if not quiet:
        stepwise_model.summary()
    return ArimaOrder(stepwise_model.order, stepwise_model.seasonal_order)

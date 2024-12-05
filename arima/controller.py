import time

from pandas import Series

import arima.data
import arima.evaluation
import arima.model
from arima.interface import ArimaOrder, OptimisationMethod


def train_and_evaluate_arima(
    dataset: Series,
    prediction_length: int,
    is_data_stationary: bool = False,
    arima_order: ArimaOrder = None,
    log_label: str = None,
    optimisation_method: str = OptimisationMethod.L_BFGS.value,
    should_find_best_order: bool = False,
    season_length: int = None,
) -> None:
    """
    Trains and evaluates an ARIMA model on the given dataset.
    Args:
        dataset (Series): The time series data to be used for training and evaluation.
        prediction_length (int): The length of the prediction.
        is_data_stationary (bool, optional): Whether the data is stationary. Used for order determination.
            Defaults to False.
        arima_order (ArimaOrder, optional): The order of the ARIMA model.
            If None, it will be determined based on the dataset. Defaults to None.
        log_label (str, optional): A label for logging purposes. Defaults to None.
        optimisation_method (str, optional): The method used for optimization.
            Defaults to OptimisationMethod.L_BFGS.value.
        should_find_best_order (bool, optional): Whether to find the best ARIMA order automatically.
            Defaults to False.
        season_length (int, optional): The length of the data seasonality.
            Defaults to None.
    Returns:
        None
    """
    find_order_runtime = None
    if should_find_best_order:
        start_time = time.time()
        arima_order = arima.data.find_best_order(
            dataset=dataset,
            season_length=season_length,
            is_data_stationary=is_data_stationary,
        )
        find_order_runtime = time.time() - start_time
    start_time = time.time()
    arima_datasets = arima.data.train_test_split_dataset(
        dataset=dataset, prediction_length=prediction_length
    )
    trained_model = arima.model.train_model(
        train_dataset=arima_datasets.train_dataset,
        arima_order=arima_order,
        optimization_method=optimisation_method,
    )
    training_runtime = time.time() - start_time
    prediction = arima.evaluation.predict(
        model=trained_model,
        arima_datasets=arima_datasets,
    )
    arima.evaluation.log(
        order=arima_order,
        prediction=prediction,
        arima_datasets=arima_datasets,
        log_label=log_label,
        training_runtime=training_runtime,
        find_order_runtime=find_order_runtime,
    )

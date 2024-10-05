import time

from pandas import Series

import arima.data
import arima.evaluation
import arima.model
from arima.interface import ArimaOrder, OptimisationMethod
from config import config


def train_and_evaluate_arima(
    dataset: Series,
    max_prediction_length: int = config.arima_max_prediction_length,
    arima_order: ArimaOrder = None,
    log_label: str = None,
    optimisation_method: str = OptimisationMethod.L_BFGS.value,
    should_find_best_order: bool = False,
    should_save_model: bool = False,
    season_length: int = None,
) -> None:
    """
    Trains and evaluates an ARIMA model using the given dataset.
    Parameters:
        dataset (pandas.Series): The time series dataset to train and evaluate the ARIMA model on.
        arima_order (ArimaOrder, optional): The order of the ARIMA model. Default is None.
        log_label (str, optional): Label of log entry.
        should_find_best_order (bool, optional): Whether to find the best order for the ARIMA model. Default is True.
        should_show_plot (bool, optional): Whether to show the plot of the predicted values. Default is True.
    Returns:
        None
    """
    if should_find_best_order:
        start_time = time.time()
        arima_order = arima.data.find_best_order(dataset, season_length=season_length)
        find_order_runtime = time.time() - start_time
    else:
        find_order_runtime = None
    start_time = time.time()
    arima_datasets = arima.data.train_test_split_dataset(
        dataset=dataset, max_prediction_length=max_prediction_length
    )
    trained_model = arima.model.train_model(
        train_dataset=arima_datasets.train_dataset,
        arima_order=arima_order,
        should_save_model=should_save_model,
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


def load_and_evaluate_arima(
    dataset: Series,
    arima_order: ArimaOrder = None,
    max_prediction_length: int = config.arima_max_prediction_length,
    log_label: str = None,
    should_show_plot: bool = True,
    should_log_prediction: bool = False,
) -> None:
    """
    Load and evaluate an ARIMA model on a given dataset.
    Parameters:
        dataset (pandas.Series): The input dataset.
        arima_order (ArimaOrder, optional): The order of the ARIMA model. Defaults to None.
        log_label (str, optional): The label for logging. Defaults to None.
        should_show_plot (bool, optional): Whether to show the plot. Defaults to True.
        should_log_prediction (bool, optional): Whether to log the prediction. Defaults to False.
    """
    trained_model = arima.model.load_model()
    arima_datasets = arima.data.train_test_split_dataset(
        dataset, max_prediction_length=max_prediction_length
    )
    arima.evaluation.predict(
        model=trained_model,
        arima_order=arima_order,
        arima_datasets=arima_datasets,
        log_label=log_label,
        should_show_plot=should_show_plot,
        should_log_prediction=should_log_prediction,
    )

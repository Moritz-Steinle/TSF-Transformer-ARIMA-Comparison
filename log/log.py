import os
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
)

from config import config


def log_prediction(
    model: Literal["ARIMA", "Transformer"],
    prediction: Series,
    training_dataset: Series,
    validation_dataset: Series,
    plot: plt.Figure = None,
    label: str = "None",
    runtimes: str = "None",
    parameters: str = "None",
) -> None:
    current_time = datetime.now()
    length_train_dataset = len(training_dataset)
    length_validation_dataset = len(validation_dataset)
    error_metrics = _calculate_error_metrics(
        validation_dataset=validation_dataset,
        trainig_dataset=training_dataset,
        prediction=prediction,
    )
    log_dataframe = DataFrame(
        {
            "label": [label],
            "error_metrics": [error_metrics],
            "runtimes": [runtimes],
            "length_train_dataset": [length_train_dataset],
            "length_validation_dataset": [length_validation_dataset],
            "parameters": [parameters],
            "prediction": [prediction],
        },
        index=[current_time],
    )
    plot.suptitle(f"{label}\n{error_metrics}")
    log_folder = _create_log_folder(model, log_label=label)
    if plot is not None:
        plot.savefig(log_folder + "/plot.png")
    log_path = f"{log_folder}/log.json"
    log_dataframe.to_json(log_path, mode="a", orient="records", lines=True)
    print(f"{model} prediction logged to {log_folder}")


def _calculate_error_metrics(
    validation_dataset: Series, trainig_dataset: Series, prediction: Series | list
) -> str:
    mae = mean_absolute_error(
        y_true=validation_dataset,
        y_pred=prediction,
    )
    smape = mean_absolute_percentage_error(
        y_true=validation_dataset, y_pred=prediction, symmetric=True
    )
    mase = mean_absolute_scaled_error(
        y_true=validation_dataset,
        y_pred=prediction,
        y_train=trainig_dataset,
    )
    return f"MAE: {mae}, SMAPE: {smape}, MASE: {mase}"


def _create_log_folder(
    model: Literal["ARIMA", "Transformer"], log_label: str = ""
) -> str:
    """
    Creates a log folder for the specified model type with a timestamp.
    Args:
        model (Literal["ARIMA", "Transformer"]): The type of model for which the log folder is being created.
                                                 It can be either "ARIMA" or "Transformer".
    Returns:
        str: The path to the created log folder.
    Raises:
        OSError: If the directory creation fails.
    """
    timestamp = _get_timestamp()
    model_path = (
        config.arima_prediction_log_path
        if model == "ARIMA"
        else config.transformer_prediction_log_path
    )
    folder_name = f"{log_label}_{timestamp}" if log_label else timestamp
    log_folder = f"{model_path}/{folder_name}"
    os.makedirs(log_folder)
    return log_folder


def _get_timestamp() -> str:
    """Helper function that returns timestamp in the format of dd-mm-yy_HH:MM:SS to avoid whitespaces
    Returns:
        str: Current timestamp
    """
    return datetime.now().strftime("%d-%m-%y_%H:%M:%S")


def get_path_with_timestamp(path: str, extension: str = None) -> str:
    """Helper function that returns a path with a timestamp to avoid overwriting files
    Args:
        path (str): The path to the file.
        extension (str, optional): The extension of the file. Defaults to None.
    Returns:
        str: The path with the timestamp.
    """
    full_path = os.path.join(path, _get_timestamp())
    if extension:
        return f"{full_path}.{extension}"
    return full_path

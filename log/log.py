import json
import os
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import DataFrame, Series
from sktime.performance_metrics.forecasting import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_error,
    median_absolute_error,
)

from config import config


def log_prediction(
    model: Literal["ARIMA", "Transformer"],
    prediction: Series,
    training_dataset: Series,
    validation_dataset: Series,
    label: str = "None",
    runtimes: str = "None",
    parameters: str = "None",
) -> None:
    current_time = datetime.now()
    length_train_dataset = len(training_dataset)
    length_validation_dataset = len(validation_dataset)
    error_metrics = _calculate_error_metrics(
        validation_dataset=validation_dataset,
        training_dataset=training_dataset,
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
    plot = _create_plot(
        validation_dataset=validation_dataset,
        training_dataset=training_dataset,
        prediction=prediction,
        log_label=label,
    )
    log_folder = _create_log_folder(model, log_label=label)
    plot.savefig(f"{log_folder}/plot.png")
    log_path = f"{log_folder}/log.json"
    log_dataframe.to_json(log_path, mode="a", orient="records", lines=True)
    print(f"{model} prediction logged to {log_folder}")


def _calculate_error_metrics(
    validation_dataset: Series, training_dataset: Series, prediction: Series | list
):
    mae = mean_absolute_error(
        y_true=validation_dataset,
        y_pred=prediction,
    )
    median_ae = median_absolute_error(
        y_true=validation_dataset,
        y_pred=prediction,
    )
    smape = mean_absolute_percentage_error(
        y_true=validation_dataset, y_pred=prediction, symmetric=True
    )
    rsme = mean_squared_error(
        y_true=validation_dataset,
        y_pred=prediction,
        square_root=True,
    )
    mase = mean_absolute_scaled_error(
        y_true=validation_dataset,
        y_pred=prediction,
        y_train=training_dataset,
    )
    error_metrics = {
        "mae": mae,
        "median_ae": median_ae,
        "smape": smape,
        "rsme": rsme,
        "mase": mase,
    }
    return error_metrics


def _create_plot(
    validation_dataset: Series,
    training_dataset: Series,
    prediction: Series | list,
    log_label: str = "",
    training_data_plot_extension: int = 200,
) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    prediction.plot(ax=ax, legend=True, linewidth=2, label="Prediction")
    prediction_length = len(validation_dataset)
    train_plot_length = prediction_length * 2 + training_data_plot_extension
    training_dataset.tail(train_plot_length).plot(ax=ax, legend=True, label="Training")
    validation_dataset.plot(ax=ax, legend=True, label="Actual", linestyle="--")
    ax.set_title(log_label)
    ax.legend()
    return fig


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

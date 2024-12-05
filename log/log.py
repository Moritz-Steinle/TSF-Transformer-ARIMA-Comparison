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

from .interface import ErrorMetrics


def log_prediction(
    model: Literal["ARIMA", "Transformer"],
    prediction: Series,
    training_dataset: Series,
    validation_dataset: Series,
    label: str = None,
    runtimes: str = None,
    parameters: str = None,
    minimal_plot: bool = False,
) -> None:
    """
    Logs the prediction results of a model along with relevant metrics and plots.
    Parameters:
        model (Literal["ARIMA", "Transformer"]): The type of model used for prediction.
        prediction (Series): The predicted values.
        training_dataset (Series): The dataset used for training the model.
        validation_dataset (Series): The dataset used for validating the model.
        label (str, optional): A label added to the timestamp name. Defaults to "None".
        runtimes (str, optional): Runtime information for the prediction. Defaults to "None".
        parameters (str, optional): Parameters used for the prediction. Defaults to "None".
        minimal_plot (bool, optional): If True, creates a minimal plot. Otherwise, creates a detailed plot. Defaults to False.
    Returns:
        None
    """
    log_folder = _create_log_folder(model, log_label=label)
    # Format and save the log data
    error_metrics = _calculate_error_metrics(
        validation_dataset=validation_dataset,
        training_dataset=training_dataset,
        prediction=prediction,
    )
    log_dataframe = DataFrame(
        {
            "label": [label] if label else [""],
            "error_metrics": [error_metrics] if error_metrics else [""],
            "runtimes": [runtimes] if runtimes else [""],
            "dataset_sizes": {
                "training": len(training_dataset),
                "validation": len(validation_dataset),
            },
            "parameters": [parameters],
            "prediction": [prediction],
        },
        index=[datetime.now()],
    )
    log_path = f"{log_folder}/log.json"
    log_dataframe.to_json(log_path, mode="a", orient="records", lines=True)
    # Create and save the plot
    plot = (
        _create_minimal_plot(
            validation_dataset=validation_dataset,
            prediction=prediction,
        )
        if minimal_plot
        else _create_plot(
            validation_dataset=validation_dataset,
            training_dataset=training_dataset,
            prediction=prediction,
            log_label=label,
        )
    )
    plot.savefig(f"{log_folder}/plot.png")
    print(f"{model} prediction logged to {log_folder}")


def _calculate_error_metrics(
    validation_dataset: Series, training_dataset: Series, prediction: Series
) -> ErrorMetrics:
    """
    Calculate various error metrics of a prediction.
    Args:
        validation_dataset (Series): The dataset used for validation.
        training_dataset (Series): The dataset used for training.
        prediction (Series): The predicted values.
    Returns:
        ErrorMetrics: An object containing the following error metrics:
            - mae: Mean Absolute Error
            - median_ae: Median Absolute Error
            - smape: Symmetric Mean Absolute Percentage Error
            - rsme: Root Mean Squared Error
            - mase: Mean Absolute Scaled Error
    """
    return ErrorMetrics(
        mae=mean_absolute_error(
            y_true=validation_dataset,
            y_pred=prediction,
        ),
        median_ae=median_absolute_error(
            y_true=validation_dataset,
            y_pred=prediction,
        ),
        smape=mean_absolute_percentage_error(
            y_true=validation_dataset, y_pred=prediction, symmetric=True
        ),
        rsme=mean_squared_error(
            y_true=validation_dataset,
            y_pred=prediction,
            square_root=True,
        ),
        mase=mean_absolute_scaled_error(
            y_true=validation_dataset,
            y_pred=prediction,
            y_train=training_dataset,
        ),
    )


def _create_plot(
    validation_dataset: Series,
    training_dataset: Series,
    prediction: Series | list,
    log_label: str = "",
    training_data_plot_extension: int = 200,
) -> Figure:
    """
    Creates a plot including the end of the
    training dataset, the validation dataset, and the prediction
    with a legend and a title.
    Args:
        validation_dataset (Series): The dataset used for validation.
        training_dataset (Series): The dataset used for training.
        prediction (Series | list): The predicted values.
        log_label (str, optional): The label for the plot title. Defaults to "".
        training_data_plot_extension (int, optional): The number of additional data points to include in the training plot. Defaults to 200.
    Returns:
        Figure: The matplotlib figure object containing the plot.
    """
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(12, 6))
    prediction.plot(ax=ax, legend=True, linewidth=2, label="Prediction")
    prediction_length = len(validation_dataset)
    train_plot_length = prediction_length * 2 + training_data_plot_extension
    training_dataset.tail(train_plot_length).plot(ax=ax, legend=True, label="Training")
    validation_dataset.plot(ax=ax, legend=True, label="Actual", linestyle="--")
    ax.set_title(log_label)
    ax.legend()
    return fig


def _create_minimal_plot(
    validation_dataset: Series,
    prediction: Series | list,
) -> Figure:
    """
    Creates a plot with the validation dataset and the prediction
    Args:
        validation_dataset (Series): The dataset used for validation.
        prediction (Series | list): The predicted values.
    Returns:
        Figure: The matplotlib figure object containing the plot.
    """
    plt.tight_layout()
    fig, ax = plt.subplots(figsize=(12, 6))
    prediction.plot(
        ax=ax,
        linewidth=2,
    )
    validation_dataset.plot(ax=ax, linestyle="--")
    return fig


def _create_log_folder(
    model: Literal["ARIMA", "Transformer"], log_label: str = ""
) -> str:
    """
    Creates a log folder for the specified model type with a timestamp.
    Args:
        model (Literal["ARIMA", "Transformer"]):
            The type of model for which the log folder is being created.
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
    """
    Helper function that returns timestamp in the format of dd-mm-yy_HH:MM:SS,
        avoiding whitespaces
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

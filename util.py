import os
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
from pandas import DataFrame, Series

from config import config


def log_prediction(
    model: Literal["ARIMA", "Transformer"],
    prediction,
    mean_squared_error: float,
    test_dataset: DataFrame | Series,
    train_dataset: DataFrame | Series,
    label: str = "None",
    runtimes: str = "None",
    parameters: str = "None",
) -> None:
    current_time = datetime.now()
    log_dataframe = DataFrame(
        {
            "label": [label],
            "mean_squared_error": [mean_squared_error],
            "runtimes": [runtimes],
            "length_train_dataset": [len(train_dataset)],
            "length_test_dataset": [len(test_dataset)],
            "parameters": [parameters],
            "prediction": [prediction.to_json()],
        },
        index=[current_time],
    )
    log_folder = _create_log_folder(model)
    log_path = f"{log_folder}/log.json"
    log_dataframe.to_json(log_path, mode="a", orient="records", lines=True)
    gen_and_save_plot(
        log_folder=log_folder,
        prediction=prediction,
        test_dataset=test_dataset,
        train_dataset=train_dataset,
        error=mean_squared_error,
    )
    print(f"{model} prediction logged to {config.arima_prediction_log_path}")


def gen_and_save_plot(
    log_folder: str,
    prediction,
    test_dataset: DataFrame | Series,
    train_dataset: DataFrame | Series,
    error: float,
) -> None:
    prediction.plot(legend=True, linewidth=2)
    prediction_length = len(test_dataset)
    train_plot_length = prediction_length * 2
    train_dataset[:train_plot_length].plot(legend=True, label="Training")
    test_dataset.plot(legend=True, label="Actual", linestyle="--")
    plt.title(f"\n MAE={error}")
    plt.savefig(log_folder + "/plot.png")


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


def _get_timestamp() -> str:
    """Helper function that returns timestamp in the format of dd-mm-yy_HH:MM:SS to avoid whitespaces
    Returns:
        str: Current timestamp
    """
    return datetime.now().strftime("%d-%m-%y_%H:%M:%S")


def _create_log_folder(model: Literal["ARIMA", "Transformer"]) -> str:
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
    log_folder = f"{model_path}/{timestamp}"
    os.makedirs(log_folder)
    return log_folder

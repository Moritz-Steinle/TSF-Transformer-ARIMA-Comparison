import os
from datetime import datetime
from typing import Literal

import matplotlib.pyplot as plt
from pandas import DataFrame

from config import config

# TODO rename to logging and move to separate folder


def log_prediction(
    model: Literal["ARIMA", "Transformer"],
    prediction: str,
    plot: plt.Figure = None,
    mean_squared_error: float = -1,
    length_test_dataset: int = -1,
    length_train_dataset: int = -1,
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
            "length_train_dataset": [length_train_dataset],
            "length_test_dataset": [length_test_dataset],
            "parameters": [parameters],
            "prediction": [prediction],
        },
        index=[current_time],
    )
    log_folder = _create_log_folder(model)
    if plot is not None:
        plot.savefig(log_folder + "/plot.png")
    log_path = f"{log_folder}/log.json"
    log_dataframe.to_json(log_path, mode="a", orient="records", lines=True)
    print(f"{model} prediction logged to {log_folder}")


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

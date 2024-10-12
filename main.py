import json
import time
from pathlib import Path

import pandas as pd
import torch
from pandas import DataFrame

import data.analyse
import data.from_db
import data.process
import transformer.data
import transformer.evaluation
import transformer.interface
import transformer.model
from arima.controller import train_and_evaluate_arima
from arima.interface import (
    ArimaOrder,
    OptimisationMethod,
    Resolution,
    get_influx_order,
    get_sawtooth_order,
)
from data.process import (
    get_influx_dataset,
    get_sawtooth_dataset,
    get_stallion_dataset,
)
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import (
    Hyperparamters,
    ModelPath,
    get_influx_hyperparameters,
    get_sawtooth_hyperparameters,
)

torch.manual_seed(0)


# Transformer
def influx_transformer():
    """
    Trains and evaluates a transformer model using the real life InfluxDB dataset.
    """
    resolution = "4h"
    max_epochs = 100
    prediction_length = 20
    hyperparameters_study_trials = 0
    log_label = f"InfluxDB_r={resolution}_e={max_epochs}_pl={prediction_length}_hst={hyperparameters_study_trials}"
    train_and_evaluate_transformer(
        dataset=get_influx_dataset(resolution=resolution),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=prediction_length
        ),
        max_epochs=max_epochs,
        log_label=log_label,
        hyperparameters=Hyperparamters(
            gradient_clip_val=6.9953515571,
            hidden_size=70,
            dropout=0.1558743686,
            hidden_continuous_size=40,
            attention_head_size=3,
            learning_rate=0.0039810717,
        ),
    )


def sawtooth_transformer_1_10():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    amount_intervals = 500
    max_epochs = 50
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(
            amount_intervals=amount_intervals,
            steps_per_interval=10,
            interval_length=10,
        ),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=36
        ),
        max_epochs=max_epochs,
        hyperparameters=Hyperparamters(
            gradient_clip_val=5.567624753786564,
            hidden_size=104,
            dropout=0.15965296238642823,
            hidden_continuous_size=41,
            attention_head_size=1,
            learning_rate=0.0031622776601683794,
        ),
        log_label=f"Sawtooth_[1,10,10]_i={amount_intervals}",
    )


def sawtooth_transformer_1_36():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    amount_intervals = 100
    max_epochs = 50
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(
            amount_intervals=amount_intervals,
            steps_per_interval=36,
            interval_length=6,
        ),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=36
        ),
        max_epochs=max_epochs,
        hyperparameters=Hyperparamters(
            gradient_clip_val=0.01565726380636608,
            hidden_size=223,
            dropout=0.21608714783830352,
            hidden_continuous_size=10,
            attention_head_size=1,
            learning_rate=0.005623413251903493,
        ),
        log_label=f"Sawtooth_[1,6,36]_i={amount_intervals}",
    )


def transformer_influx_resolution_comparison():
    """
    Trains and evaluates ARIMA models using the Influx data with different optimization methods and resolutions.
    Logs all results.
    """
    resolutions = ["12h", "8h", "6h", "5h", "4h", "3h", "2h"]
    max_epochs = 100
    prediction_length = 35
    for resolution in resolutions:
        train_and_evaluate_transformer(
            dataset=get_influx_dataset(resolution=resolution),
            dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
                max_prediction_length=prediction_length
            ),
            max_epochs=max_epochs,
            hyperparameters=get_influx_hyperparameters(),
            log_label=(f"Influx_i={resolution}"),
        )
        time.sleep(60)


def tutorial_transformer():
    """
    Trains and evaluates a transformer model using the dataset from the pytorch_forecasting tutorial.
    (https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
    """
    train_and_evaluate_transformer(
        dataset=get_stallion_dataset(),
        dataloader_parameters=transformer.interface.get_stallion_dataset_parameters(),
        max_epochs=1,
        hyperparameters=transformer.interface.get_stallion_hyperparameters(),
        fast_dev_run=True,
    )


def evaluate_saved_transformer(dataset: DataFrame, model_path: ModelPath = None):
    """
    Evaluates a saved transformer model.
    Args:
        dataset (DataFrame): The dataset to evaluate the model on.
        model_path (ModelPath, optional): The path to the saved model. Defaults to None.
            If none is given, the path in transformer/best_model_path.txt is used.
    """
    dataloaders = transformer.data.create_dataloaders(dataset)
    transformer.evaluation.make_prediction(
        model=transformer.model.load_model(model_path),
        val_dataloader=dataloaders.val_dataloader,
    )


# ARIMA
def influx_arima():
    """
    Trains and evaluates an ARIMA model using the real life InfluxDB dataset.
    """
    resolution = "4h"
    optimization_method = OptimisationMethod.L_BFGS.value
    arima_order = ArimaOrder(order=(0, 0, 2), seasonal_order=(2, 0, 2, 43))
    # log_label = f"InfluxDB_r={resolution}_om={optimization_method}_order={arima_order}"
    log_label = f"Influx DB, 4h resolution, 1 season chained"
    train_and_evaluate_arima(
        dataset=get_influx_dataset(
            resolution=resolution,
        )["value"],
        max_prediction_length=25,
        log_label=log_label,
        optimisation_method=optimization_method,
        arima_order=arima_order,
    )


def sawtooth_arima():
    """
    Trains and evaluates an ARIMA model using a sawtooth function dataset.
    """
    amount_intervals = 100
    dataset = get_sawtooth_dataset(
        amount_intervals=amount_intervals, steps_per_interval=36, interval_length=6
    )["value"]
    season_length = 30
    # arima_order = ArimaOrder(order=(4, 0, 1), seasonal_order=(2, 0, 0, season_length))
    arima_order = ArimaOrder(order=(2, 0, 0), seasonal_order=(2, 0, 2, season_length))
    train_and_evaluate_arima(
        dataset=dataset,
        log_label=f"Sawtooth_i={amount_intervals}",
        # arima_order=arima_order,
        max_prediction_length=36,
        arima_order=arima_order,
    )


def arima_resolution_comparison():
    """
    Trains and evaluates ARIMA models using the Influx data with different optimization methods and resolutions.
    Logs all results.
    """
    amounts_interval = [10, 20, 30, 40, 50, 100]
    optimisation_method = OptimisationMethod.L_BFGS.value
    steps_per_interval = 10
    interval_length = 10
    max_prediction_length = 36
    arima_order = ArimaOrder(order=(4, 0, 1), seasonal_order=(2, 0, 0, 9))
    for amount_interval in amounts_interval:
        train_and_evaluate_arima(
            dataset=get_sawtooth_dataset(
                amount_intervals=amount_interval,
                steps_per_interval=steps_per_interval,
                interval_length=interval_length,
            )["value"],
            log_label=f"Sawtooth_[1,{interval_length},{steps_per_interval}]_i={amount_interval}",
            max_prediction_length=max_prediction_length,
            optimisation_method=optimisation_method,
            arima_order=arima_order,
        )


# Util
def fetch_data_from_db():
    """
    Fetches data from the database.
    Resolution sets the time interval of the data.
    """
    resolutions = ["5m"]
    for resolution in resolutions:
        data.from_db.fetch(resolution)


def plot_dataset():
    """
    Plots the "value" column of a dataset.
    """
    dataset = get_influx_dataset(resolution="2h")
    data.analyse.plot_dataset(dataset=dataset)


def analyse_dataset():
    """
    Analyses the "value" column of a dataset.
    """
    dataset = get_influx_dataset(resolution="10s", should_normalize=False)
    data.analyse.analyse_dataset(dataset=dataset)


# Generated by Claude.ai
def collect_logs():
    folder_name = "Sawtooth[1,36] 100 epochs"
    # Base path for the ARIMA results
    base_path = Path(f"transformer/results/{folder_name}")

    all_data = []

    pd.set_option("display.float_format", "{:.10f}".format)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 150)

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            log_file = subdir / "log.json"
            if log_file.exists():
                with open(log_file, "r") as f:
                    log_data = json.load(f)

                # Extract only the required fields
                extracted_data = {
                    "Label": log_data["label"],
                    "MAE": log_data["error_metrics"]["mae"],
                    "MASE": log_data["error_metrics"]["mase"],
                    "Runtime": log_data["runtimes"],
                }

                all_data.append(extracted_data)

    # Create a DataFrame from all the collected data
    df = pd.DataFrame(all_data)
    return df


print(collect_logs())

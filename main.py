import time

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
    resolution = "4h-1_season_chained_large"
    max_epochs = 50
    prediction_length = 25
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
            gradient_clip_val=17.149865385362546,
            hidden_size=42,
            dropout=0.22085830299991055,
            hidden_continuous_size=24,
            attention_head_size=1,
            learning_rate=0.019952623149688802,
        ),
    )


def fast_dev_transformer():
    """
    Trains and evaluates a transformer model using the real life InfluxDB dataset.
    """
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(amount_intervals=10),
        max_epochs=1,
        fast_dev_run=True,
    )


def sawtooth_transformer():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    amount_intervals = 1000
    max_epochs = 1
    prediction_length = 36
    dataset = get_sawtooth_dataset(
        amount_intervals=amount_intervals, steps_per_interval=36, interval_length=6
    )
    log_label = f"Sawtooth_i={amount_intervals}_e={max_epochs}_pl={prediction_length}"
    train_and_evaluate_transformer(
        dataset=dataset,
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=prediction_length
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
        log_label=log_label,
    )


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
    resolution = "4h-1_season_chained"
    optimization_method = OptimisationMethod.L_BFGS.value
    # arima_order = ArimaOrder(order=(0, 0, 2), seasonal_order=(2, 0, 2, 84))
    arima_order = ArimaOrder(order=(0, 1, 0), seasonal_order=(0, 0, 0, 43))
    # log_label = f"InfluxDB_r={resolution}_om={optimization_method}_order={arima_order}"
    log_label = f"InfluxDB_r={resolution}_om={optimization_method}"
    train_and_evaluate_arima(
        dataset=get_influx_dataset(
            resolution=resolution,
        )["value"],
        max_prediction_length=200,
        log_label=log_label,
        optimisation_method=optimization_method,
        arima_order=arima_order,
        # should_find_best_order=True,
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
        max_prediction_length=35,
        arima_order=arima_order,
    )


def arima_method_resolution_comparison():
    """
    Trains and evaluates ARIMA models using the Influx data with different optimization methods and resolutions.
    Logs all results.
    """
    amounts_interval = [3, 5, 10, 20, 40, 70, 100]
    optimisation_method = OptimisationMethod.L_BFGS.value
    dataset = get_sawtooth_dataset(
        amount_intervals=10, steps_per_interval=36, interval_length=6
    )["value"]
    arima_order = ArimaOrder(order=(4, 0, 1), seasonal_order=(2, 0, 0, 30))

    for amount_interval in amounts_interval:
        train_and_evaluate_arima(
            dataset=dataset,
            log_label=f"Sawtooth_i={amount_interval}_om={optimisation_method}",
            max_prediction_length=25,
            optimisation_method=optimisation_method,
            arima_order=arima_order,
        )


# Util
def fetch_data_from_db():
    """
    Fetches data from the database.
    Resolution sets the time interval of the data.
    """
    resolution = "2h"
    data.from_db.fetch(resolution)


def plot_dataset():
    """
    Plots the "value" column of a dataset.
    """
    dataset = get_sawtooth_dataset(
        amount_intervals=10, steps_per_interval=36, interval_length=6
    )
    print(dataset)
    data.analyse.plot_dataset(dataset=dataset)


def analyse_dataset():
    """
    Analyses the "value" column of a dataset.
    """
    dataset = get_influx_dataset(resolution="10s", should_normalize=False)
    data.analyse.analyse_dataset(dataset=dataset)


sawtooth_arima()

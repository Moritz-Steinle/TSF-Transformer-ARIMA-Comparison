import matplotlib.pyplot as plt
import torch

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
)
from data.process import (
    get_influx_dataset,
    get_sawtooth_dataset,
)
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import Hyperparamters

torch.manual_seed(0)


# Transformer
def trasformer_empirical():
    """
    Trains and evaluates a transformer model using the real life InfluxDB dataset.
    """
    resolution = "12h_chained"
    max_epochs = 200
    prediction_length = 14
    hyperparameters_study_trials = 50
    log_label = f"InfluxDB_r={resolution}_e={max_epochs}_pl={prediction_length}_hst={hyperparameters_study_trials}"
    train_and_evaluate_transformer(
        dataset=get_influx_dataset(resolution=resolution),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=prediction_length
        ),
        max_epochs=max_epochs,
        log_label=log_label,
        hyperparameters_study_trials=hyperparameters_study_trials,
        hyperparameters=Hyperparamters(
            gradient_clip_val=6.9953515571,
            hidden_continuous_size=40,
            dropout=0.1558743686,
            attention_head_size=3,
            learning_rate=0.0039810717,
            hidden_size=0,
        ),
    )


def transformer_sawtooth_coarse():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    amount_intervals = 500
    max_epochs = 50
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(
            amount_intervals=amount_intervals,
            steps_per_interval=10,
            max_value=10,
        ),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=36
        ),
        max_epochs=max_epochs,
        hyperparameters=Hyperparamters(
            gradient_clip_val=33.134465883069915,
            hidden_size=31,
            dropout=0.13584274723675938,
            hidden_continuous_size=20,
            attention_head_size=2,
            learning_rate=0.022387211385683406,
        ),
        log_label=f"Sawtooth_[1,10,10]_i={amount_intervals}",
    )


def transformer_sawtooth_fine():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    amount_intervals = 400
    max_epochs = 50
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(
            amount_intervals=amount_intervals,
            steps_per_interval=30,
            max_value=6,
        ),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=36
        ),
        max_epochs=max_epochs,
        hyperparameters=Hyperparamters(
            gradient_clip_val=0.01565726380636608,
            hidden_size=0,
            dropout=0.21608714783830352,
            hidden_continuous_size=10,
            attention_head_size=1,
            learning_rate=0.005623413251903493,
        ),
        log_label=f"Sawtooth_[1,6,36]_i={amount_intervals}",
    )


# ARIMA
def arima_empirical():
    """
    Trains and evaluates an ARIMA model using the real life InfluxDB dataset.
    """
    resolution = "12h"
    season_length = 7
    max_prediction_length = 14
    should_find_best_order = False
    arima_order = ArimaOrder(order=(0, 0, 2), seasonal_order=(2, 0, 2, season_length))
    optimization_method = OptimisationMethod.L_BFGS.value
    log_label = f"{resolution}"
    train_and_evaluate_arima(
        dataset=get_influx_dataset(
            resolution=resolution,
        )["value"],
        max_prediction_length=max_prediction_length,
        log_label=log_label,
        optimisation_method=optimization_method,
        arima_order=arima_order,
        should_find_best_order=should_find_best_order,
    )


def arima_synthetic_coarse():
    """
    Trains and evaluates an ARIMA model using a coarse sawtooth function dataset.
    """
    amount_intervals = 500
    max_prediction_length = 36
    season_length = 10
    max_value = 10
    dataset = get_sawtooth_dataset(
        amount_intervals=amount_intervals,
        steps_per_interval=season_length,
        max_value=max_value,
    )["value"]
    arima_order = ArimaOrder(order=(2, 0, 1), seasonal_order=(2, 0, 2, season_length))
    train_and_evaluate_arima(
        dataset=dataset,
        log_label=f"Sawtooth [1,36] i={amount_intervals}",
        max_prediction_length=max_prediction_length,
        arima_order=arima_order,
    )


def arima_synthetic_fine():
    """
    Trains and evaluates an ARIMA model using a coarse sawtooth function dataset.
    """
    amount_intervals = 500
    max_prediction_length = 36
    season_length = 30
    max_value = 6
    dataset = get_sawtooth_dataset(
        amount_intervals=amount_intervals,
        steps_per_interval=season_length,
        max_value=max_value,
    )["value"]
    arima_order = ArimaOrder(order=(2, 0, 1), seasonal_order=(2, 0, 2, season_length))
    train_and_evaluate_arima(
        dataset=dataset,
        log_label=f"Sawtooth [1,36] i={amount_intervals}",
        max_prediction_length=max_prediction_length,
        arima_order=arima_order,
    )


# Util
def fetch_data_from_db():
    """
    Fetches data from the database.
    Resolution sets the time interval of the data.
    """
    resolutions = ["10s"]
    for resolution in resolutions:
        data.from_db.fetch(resolution)


def plot_dataset():
    """
    Plots the "value" column of a dataset.
    """
    dataset = get_influx_dataset(
        resolution="4h-1-season-chained", should_normalize=False
    ).tail(200)
    data.analyse.plot_dataset(dataset=dataset["value"])


def analyse_dataset():
    """
    Analyses the "value" column of a dataset.
    """
    dataset = get_influx_dataset(resolution="2h", should_normalize=False)
    data.analyse.analyse_dataset(dataset=dataset)


plot_dataset()

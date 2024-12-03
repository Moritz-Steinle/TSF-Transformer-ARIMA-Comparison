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
    get_csv_dataset,
    get_sawtooth_dataset,
)
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import Hyperparamters

# Set seed for reproducibility
torch.manual_seed(0)


# Transformer
def transformer_empirical():
    """
    Trains and evaluates Transformer using an empirical dataset.
    """
    train_and_evaluate_transformer(
        max_training_epochs=1,
        hyperparameters_study_trials=0,
        dataset=get_csv_dataset(filename="12h_chained"),
        dataloader_parameters=transformer.interface.get_base_dataloader_parameters(
            max_prediction_length=14
        ),
        hyperparameters=Hyperparamters(
            gradient_clip_val=6.9953515571,
            hidden_continuous_size=40,
            dropout=0.1558743686,
            attention_head_size=3,
            learning_rate=0.0039810717,
            hidden_size=0,
        ),
    )


def transformer_synthetic():
    """
    Trains and evaluates Transformer using a synthetic dataset based on a sawtooth function.
    """
    train_and_evaluate_transformer(
        max_training_epochs=1,
        dataset=get_sawtooth_dataset(
            amount_intervals=500,
            steps_per_interval=10,
            max_value=10,
        ),
        dataloader_parameters=transformer.interface.get_base_dataloader_parameters(
            max_prediction_length=36
        ),
        hyperparameters=Hyperparamters(
            gradient_clip_val=33.134465883069915,
            hidden_size=31,
            dropout=0.13584274723675938,
            hidden_continuous_size=20,
            attention_head_size=2,
            learning_rate=0.022387211385683406,
        ),
    )


# ARIMA
def arima_empirical():
    """
    Trains and evaluates ARIMA using an empirical dataset.
    """
    resolution = "12h"
    season_length = 7
    max_prediction_length = 14
    should_find_best_order = False
    arima_order = ArimaOrder(order=(0, 0, 2), seasonal_order=(2, 0, 2, season_length))
    optimization_method = OptimisationMethod.L_BFGS.value
    log_label = f"{resolution}"
    train_and_evaluate_arima(
        dataset=get_csv_dataset(
            filename=resolution,
        )["value"],
        max_prediction_length=max_prediction_length,
        log_label=log_label,
        optimisation_method=optimization_method,
        arima_order=arima_order,
        should_find_best_order=should_find_best_order,
    )


def arima_synthetic():
    """
    Trains and evaluates ARIMA using a synthetic dataset based on a sawtooth function.
    """
    season_length = 10
    train_and_evaluate_arima(
        dataset=get_sawtooth_dataset(
            amount_intervals=500,
            steps_per_interval=season_length,
            max_value=10,
        )["value"],
        max_prediction_length=36,
        arima_order=ArimaOrder(
            order=(2, 0, 1), seasonal_order=(2, 0, 2, season_length)
        ),
    )


# Util
def plot_series():
    """
    Example for plotting a series of data
    """
    dataset = get_csv_dataset(
        filename="4h-1-season-chained", should_normalize=False
    ).tail(200)
    data.analyse.plot_series(series=dataset["value"])


def analyse_series():
    """
    Example for analysing a series of data
    """
    dataset = get_csv_dataset(
        filename="2h",
        should_fill_missing=False,
        should_normalize=False,
    )
    filled_dataset = get_csv_dataset(
        filename="2h",
    )
    data.analyse.analyse_series(
        raw_series=dataset["value"], filled_series=filled_dataset["value"]
    )


analyse_series()

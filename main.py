from pandas import DataFrame

import data.from_db
import transformer.data
import transformer.evaluation
import transformer.interface
import transformer.model
from arima.controller import train_and_evaluate_arima
from arima.interface import get_sawtooth_order
from data.process import get_influx_dataset, get_sawtooth_dataset, get_stallion_dataset
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import (
    ModelPath,
    get_influx_hyperparameters,
    get_sawtooth_hyperparameters,
)


# Transformer
def influx_transformer():
    train_and_evaluate_transformer(
        dataset=get_influx_dataset(),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(),
        hyperparameters=get_influx_hyperparameters(),
        fast_dev_run=True,
    )


def sawtooth_transformer():
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(amount_interval=1000),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(),
        hyperparameters=get_sawtooth_hyperparameters(),
        should_run_hyperparameter_study=False,
    )


def tutorial_transformer():
    train_and_evaluate_transformer(
        dataset=get_stallion_dataset(),
        dataloader_parameters=transformer.interface.get_stallion_dataset_parameters(),
        hyperparameters=transformer.interface.get_stallion_hyperparameters(),
    )


def evaluate_saved_transformer(dataset: DataFrame, model_path: ModelPath = None):
    dataloaders = transformer.data.create_dataloaders(dataset)
    transformer.evaluation.make_prediction(
        model=transformer.model.load_model(model_path),
        val_dataloader=dataloaders.val_dataloader,
    )


# ARIMA
def influx_arima():
    resolution = "8h"
    train_and_evaluate_arima(
        dataset=get_influx_dataset(resolution=resolution)["value"],
        should_find_best_order=False,
        log_label="InfluxDB",
        should_show_plot=False,
    )


def sawtooth_arima():
    resolution = "8h"
    train_and_evaluate_arima(
        dataset=get_sawtooth_dataset(amount_interval=1000)["value"],
        log_label="Sawtooth",
        arima_order=get_sawtooth_order(resolution),
        should_show_plot=True,
        should_find_best_order=False,
        should_save_model=False,
    )


def run_arima_comparison():
    resolutions = ["24h", "12h", "8h", "6h"]
    for resolution in resolutions:
        data.from_db.fetch(resolution)
        dataset = get_influx_dataset()["value"]
        train_and_evaluate_arima(
            dataset=dataset,
            log_label=f"FluxDB_{resolution}",
        )


# Util
def fetch_data_from_db():
    resolution = "24h"
    data.from_db.fetch(resolution)


sawtooth_arima()

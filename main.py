from pandas import DataFrame

import arima.data
import arima.interface
import data.analyse
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
        dataset=get_influx_dataset(resolution="8h", should_normalize=False),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=6
        ),
        hyperparameters=get_influx_hyperparameters(),
        fast_dev_run=True,
    )


def sawtooth_transformer():
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(amount_interval=1000),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=150
        ),
        hyperparameters=get_sawtooth_hyperparameters(),
        fast_dev_run=True,
    )


def tutorial_transformer():
    train_and_evaluate_transformer(
        dataset=get_stallion_dataset(),
        dataloader_parameters=transformer.interface.get_stallion_dataset_parameters(),
        hyperparameters=transformer.interface.get_stallion_hyperparameters(),
        fast_dev_run=True,
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
        arima_order=arima.interface.get_influx_order(resolution),
        log_label="InfluxDB",
    )


def sawtooth_arima():
    train_and_evaluate_arima(
        dataset=get_sawtooth_dataset(amount_interval=1000)["value"],
        arima_order=get_sawtooth_order(),
        log_label="Sawtooth",
    )


def run_arima_comparison():
    resolutions = ["24h", "12h", "8h", "6h"]
    for resolution in resolutions:
        dataset = get_influx_dataset(resolution=resolution)["value"]
        train_and_evaluate_arima(
            dataset=dataset,
            log_label=f"FluxDB_{resolution}",
            max_prediction_length=6,
            arima_order=arima.interface.get_influx_order(resolution=resolution),
        )


# Util
def fetch_data_from_db():
    """
    Fetches data from the database.
    Resolution sets the time interval of the data.
    """
    resolution = "24h"
    data.from_db.fetch(resolution)


influx_transformer()

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
from arima.interface import ArimaOrder, get_sawtooth_order
from data.process import get_influx_dataset, get_sawtooth_dataset, get_stallion_dataset
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import (
    ModelPath,
    get_influx_hyperparameters,
    get_sawtooth_hyperparameters,
)

# TODO save plot to file
# TODO add runtime to transfomer
# TODO fix arimawarnings
# TODO remove prediction length from config


# Transformer
def influx_transformer():
    """
    Trains and evaluates a transformer model using the real life InfluxDB dataset.
    """
    train_and_evaluate_transformer(
        dataset=get_influx_dataset(resolution="6h"),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=20
        ),
        should_run_hyperparameter_study=True,
    )


def sawtooth_transformer():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(amount_intervals=1000),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=150
        ),
        hyperparameters=get_sawtooth_hyperparameters(),
        fast_dev_run=True,
    )


def tutorial_transformer():
    """
    Trains and evaluates a transformer model using the dataset from the pytorch_forecasting tutorial.
    (https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html)
    """
    train_and_evaluate_transformer(
        dataset=get_stallion_dataset(),
        dataloader_parameters=transformer.interface.get_stallion_dataset_parameters(),
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
    resolution = "8h"
    train_and_evaluate_arima(
        dataset=get_influx_dataset(resolution=resolution)["value"],
        arima_order=arima.interface.get_influx_order(resolution),
        log_label="InfluxDB",
    )


def sawtooth_arima():
    """
    Trains and evaluates an ARIMA model using a sawtooth function dataset.
    """
    train_and_evaluate_arima(
        dataset=get_sawtooth_dataset(amount_intervals=10)["value"],
        log_label="Sawtooth",
        arima_order=ArimaOrder(order=(4, 0, 1), seasonal_order=(2, 0, 0, 9)),
    )


def run_arima_comparison():
    """
    Trains and evaluates ARIMA models using the real life InfluxDB dataset with different resolutions.
    Logs all results.
    """
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


sawtooth_arima()

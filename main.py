from pandas import DataFrame

import data.from_db
import transformer.data
import transformer.evaluation
import transformer.interface
import transformer.model
from arima.controller import train_and_evaluate_arima
from data.process import get_influx_dataset, get_sawtooth_dataset, get_stallion_dataset
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import (
    ModelPath,
    get_influx_hyperparameters,
    get_sawtooth_hyperparameters,
)


def influx_transformer():
    train_and_evaluate_transformer(
        dataset=get_influx_dataset(),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(),
        hyperparameters=get_influx_hyperparameters(),
    )


def sawtooth_transformer():
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(amount_interval=1000),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(),
        hyperparameters=get_sawtooth_hyperparameters(),
        should_run_hyperparameter_study=False,
        fast_dev_run=True,
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


def influx_arima():
    train_and_evaluate_arima(
        dataset=get_influx_dataset()["value"],
        max_prediction_length=20,
        should_find_best_order=True,
        log_label="InfluxDB",
        should_show_plot=False,
    )


def sawtooth_arima():
    train_and_evaluate_arima(
        dataset=get_sawtooth_dataset(amount_interval=1000)["value"],
        max_prediction_length=20,
        should_find_best_order=True,
        log_label="Sawtooth",
        should_show_plot=True,
        should_save_model=False,
    )


def run_arima_comparison():
    sample_intervals = ["24h", "12h", "8h", "6h"]
    for sample_interval in sample_intervals:
        data.from_db.fetch(sample_interval)
        dataset = get_influx_dataset()["value"]
        train_and_evaluate_arima(
            dataset=dataset,
            max_prediction_length=20,
            log_label=f"FluxDB_{sample_interval}",
        )


sawtooth_arima()

from pandas import DataFrame

import arima.data
import arima.evaluation
import arima.model
import data.from_db
import transformer.data
import transformer.evaluation
import transformer.interface
import transformer.model
from arima.interface import ArimaOrder
from config import config
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


### ARIMA ###


def get_arima_stats(dataset):
    arima.data.plot_acf_pacf(dataset)
    print(arima.data.get_p_value(dataset))


def find_best_order(dataset) -> ArimaOrder:
    return arima.data.find_best_order(
        dataset,
        quiet=False,
    )


def train_and_evaluate_arima(log_label: str = None):
    dataset = get_influx_dataset()["value"]
    arima_order = find_best_order(dataset)
    arima_datasets = arima.data.train_test_split_dataset(
        dataset, config.max_prediction_length
    )
    trained_model = arima.model.train_model(arima_datasets.train_dataset, arima_order)
    arima.evaluation.predict(
        should_show_plot=True,
        model=trained_model,
        arima_order=arima_order,
        arima_datasets=arima_datasets,
        log_label=log_label,
    )


def load_and_evaluate_arima(dataset, arima_order):
    trained_model = arima.model.load_model()
    arima_datasets = arima.data.train_test_split_dataset(
        dataset, config.max_prediction_length
    )
    arima.evaluation.predict(
        model=trained_model,
        arima_order=arima_order,
        arima_datasets=arima_datasets,
        log_label="Sawtooth",
        should_show_plot=True,
        should_log_prediction=False,
    )


def run_arima_comparison():
    sample_intervals = ["24h", "12h", "8h", "6h"]

    for sample_interval in sample_intervals:
        data.from_db.fetch(sample_interval)
        train_and_evaluate_arima(f"FluxDB_{sample_interval}")


sawtooth_transformer()

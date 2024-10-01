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
    OptimizationMethod,
    Resolution,
    get_influx_order,
    get_sawtooth_order,
)
from data.process import (
    get_influx_dataset,
    get_sawtooth_dataset,
    get_stallion_dataset,
    get_stock_price_dataset,
)
from transformer.controller import train_and_evaluate_transformer
from transformer.interface import (
    Hyperparamters,
    ModelPath,
    get_influx_hyperparameters,
    get_sawtooth_hyperparameters,
)

# TODO
# Add prediction directly after drop
# TODO(Optional)
# Add hyperparameter study to logging folder
# fix arima warnings
# add logging to loaded models
# limit transformer output nodes to 1
# Update interface hyperparameters with study results


# Transformer
def influx_transformer():
    """
    Trains and evaluates a transformer model using the real life InfluxDB dataset.
    """
    resolution = "8h"
    max_epochs = 1
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
        hyperparameters=get_influx_hyperparameters(),
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
    amount_intervals = 375
    max_epochs = 1
    prediction_length = 15
    log_label = f"Sawtooth_i={amount_intervals}_e={max_epochs}_pl={prediction_length}"
    train_and_evaluate_transformer(
        dataset=get_sawtooth_dataset(amount_intervals=amount_intervals),
        dataloader_parameters=transformer.interface.get_influx_dataloader_parameters(
            max_prediction_length=prediction_length
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
    optimization_method = OptimizationMethod.L_BFGS.value
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
        optimization_method=optimization_method,
        arima_order=arima_order,
        # should_find_best_order=True,
    )


def sawtooth_arima():
    """
    Trains and evaluates an ARIMA model using a sawtooth function dataset.
    """
    train_and_evaluate_arima(
        dataset=get_sawtooth_dataset(amount_intervals=100)["value"],
        log_label="Sawtooth",
        arima_order=ArimaOrder(order=(4, 0, 1), seasonal_order=(2, 0, 0, 9)),
    )


# TODO remove
def stock_price_arima():
    """
    Trains and evaluates an ARIMA model using the stock price dataset.
    """
    train_and_evaluate_arima(
        dataset=get_stock_price_dataset(),
        log_label="StockPrice",
        max_prediction_length=20,
        should_find_best_order=True,
        # arima_order=ArimaOrder(order=(1, 1, 2), seasonal_order=(0, 0, 0, 0)),
    )


def arima_method_resolution_comparison():
    """
    Trains and evaluates ARIMA models using the Influx data with different optimization methods and resolutions.
    Logs all results.
    """
    optimization_methods = [
        # OptimizationMethod.BFGS.value,
        # OptimizationMethod.L_BFGS.value,
        # OptimizationMethod.CG.value,
        # OptimizationMethod.NCG.value,
        OptimizationMethod.POWELL.value,
    ]
    resolutions = [
        Resolution.H24.value,
        Resolution.H12.value,
        Resolution.H8.value,
        Resolution.H6.value,
        Resolution.H4.value,
        Resolution.H3.value,
        # Resolution.H2.value,
        Resolution.H12_CHAINED.value,
    ]
    for optimization_method in optimization_methods:
        for resolution in resolutions:
            print(
                f"Optimization method: {optimization_method}, Resolution: {resolution}"
            )
            dataset = get_influx_dataset(resolution=resolution)["value"]
            train_and_evaluate_arima(
                dataset=dataset,
                log_label=f"InfluxDB_r={resolution}_om={optimization_method}",
                max_prediction_length=6,
                optimization_method=optimization_method,
                arima_order=get_influx_order(resolution),
            )


# Util
def fetch_data_from_db():
    """
    Fetches data from the database.
    Resolution sets the time interval of the data.
    """
    resolution = "10s"
    data.from_db.fetch(resolution)


def plot_dataset():
    """
    Plots the "value" column of a dataset.
    """
    dataset = get_sawtooth_dataset(amount_intervals=10)
    data.analyse.plot_dataset(dataset=dataset)


def analyse_dataset():
    """
    Analyses the "value" column of a dataset.
    """
    dataset = get_influx_dataset(resolution="10s", should_normalize=False)
    data.analyse.analyse_dataset(dataset=dataset)


sawtooth_transformer()

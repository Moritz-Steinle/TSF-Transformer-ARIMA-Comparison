from pandas import DataFrame

import arima.data
import arima.interface
import data.analyse
import data.from_db
import data.process
import transformer.data
import transformer.evaluation
import transformer.interface
import transformer.model
from arima.controller import train_and_evaluate_arima
from arima.interface import ArimaOrder, get_sawtooth_order
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
    resolution = "2h-chained_atDrop_XXL"
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


def sawtooth_transformer():
    """
    Trains and evaluates a transformer model using a sawtooth function dataset.
    """
    amount_intervals = 312
    max_epochs = 100
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
    resolution = "12h_chained"
    log_label = f"InfluxDB_r={resolution}"
    train_and_evaluate_arima(
        dataset=get_influx_dataset(resolution=resolution)["value"],
        max_prediction_length=80,
        log_label=log_label,
        should_find_best_order=True,
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
    resolution = "10s"
    data.from_db.fetch(resolution)


def plot_dataset():
    """
    Plots the "value" column of a dataset.
    """
    dataset = get_stock_price_dataset()
    data.analyse.plot_dataset(dataset=dataset, key="High")


def analyse_dataset():
    """
    Analyses the "value" column of a dataset.
    """
    dataset = get_influx_dataset(resolution="10s", should_normalize=False)
    data.analyse.analyse_dataset(dataset=dataset)


stock_price_arima()

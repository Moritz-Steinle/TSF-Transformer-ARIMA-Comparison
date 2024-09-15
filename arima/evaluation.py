from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import config

from .interface import ArimaDatasets, ArimaOrder


def predict(
    model: SARIMAX,
    arima_datasets: ArimaDatasets,
    arima_order: ArimaOrder = None,
    should_log_prediction: bool = True,
    log_label=None,
    should_print_predictions: bool = False,
    should_show_plot: bool = False,
    runtime: int = None,
    find_order_runtime: int = None,
):
    train_dataset = arima_datasets.train_dataset
    test_dataset = arima_datasets.test_dataset
    start = len(train_dataset)
    end = len(train_dataset) + len(test_dataset) - 1
    prediction = model.predict(start=start, end=end, typ="linear").rename(
        "ARIMA predictions"
    )
    error = np.sqrt(mean_squared_error(test_dataset, prediction))
    if should_print_predictions:
        print("ARIMA predictions:")
        print(f"Mean Squared Error: {error}")
        print(prediction)
    if should_show_plot:
        prediction.plot(legend=True, linewidth=2)
        train_dataset.plot(legend=True, label="Training")
        test_dataset.plot(legend=True, label="Actual", linestyle="--")
        plt.title(f"ARIMA Prediction \n MAE={error}")
        plt.show()
    if should_log_prediction:
        log_prediction(
            arima_order=arima_order,
            arima_datasets=arima_datasets,
            prediction=prediction,
            label=log_label,
            mean_squared_error=error,
            runtime=runtime,
            find_order_runtime=find_order_runtime,
        )
    return prediction


def log_prediction(
    prediction,
    mean_squared_error: float,
    arima_datasets: ArimaDatasets = None,
    arima_order: ArimaOrder = None,
    label: str = None,
    runtime: int = None,
    find_order_runtime: int = None,
) -> None:
    order = arima_order.order if arima_order else "missing"
    seasonal_order = arima_order.seasonal_order if arima_order else "missing"
    runtime = runtime if runtime else "missing"
    find_order_runtime = find_order_runtime if find_order_runtime else "missing"
    current_time = datetime.now()
    log_dataframe = DataFrame(
        {
            "label": [label],
            "order": [order],
            "seasonal_order": [seasonal_order],
            "mean_squared_error": [mean_squared_error],
            "runtime": [runtime, find_order_runtime],
            "length_train_dataset": [len(arima_datasets.train_dataset)],
            "length_test_dataset": [len(arima_datasets.test_dataset)],
            "prediction": [prediction.to_json()],
        },
        index=[current_time],
    )
    log_dataframe.to_json(
        config.arima_prediction_log_path, mode="a", orient="records", lines=True
    )
    print(f"ARIMA Prediction logged to {config.arima_prediction_log_path}")

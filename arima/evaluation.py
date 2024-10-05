from matplotlib import pyplot
from matplotlib.figure import Figure
from pandas import Series
from statsmodels.tsa.statespace.sarimax import SARIMAX

from log.log import log_prediction

from .interface import ArimaDatasets, ArimaOrder


def predict(
    model: SARIMAX,
    arima_datasets: ArimaDatasets,
) -> Series:
    """
    Generate predictions using a SARIMAX model on provided ARIMA datasets.
    Args:
        model (SARIMAX): The SARIMAX model used for generating predictions.
        arima_datasets (ArimaDatasets): An object containing the training and test datasets.
    Returns:
        pandas.Series: The predicted values for the test dataset.
    """
    train_dataset = arima_datasets.train_dataset
    start = len(train_dataset)
    end = len(train_dataset) + len(arima_datasets.validation_dataset) - 1
    prediction = model.predict(start=start, end=end, typ="linear")
    return Series(prediction)


def log(
    order: ArimaOrder,
    prediction: Series,
    arima_datasets: ArimaDatasets,
    training_runtime: float,
    log_label: str = None,
    find_order_runtime=None,
):
    """
    Logs the performance and parameters of an ARIMA model.
    Parameters:
        model (SARIMAX): The trained SARIMAX model.
        prediction: The predicted values from the model.
        arima_datasets (ArimaDatasets): The datasets used for training and testing the model.
        training_runtime (float): The time taken to train the model.
        log_label (str, optional): An optional label for the log entry. Defaults to None.
        find_order_runtime (float, optional): The time taken to find the optimal order for the model. Defaults to None.
    Returns:
        None
    """
    parameters = f"order={order.order}, seasonal_order={order.seasonal_order}"
    runtime_string = f"Training: {training_runtime:.2f} seconds"
    if find_order_runtime is not None:
        runtime_string += f" , order study: {find_order_runtime:.2f} seconds)"
    plot = _create_plot(
        prediction=prediction,
        arima_datasets=arima_datasets,
        log_label=log_label,
    )
    log_prediction(
        model="ARIMA",
        prediction=prediction,
        training_dataset=arima_datasets.train_dataset,
        validation_dataset=arima_datasets.validation_dataset,
        plot=plot,
        label=log_label,
        runtimes=runtime_string,
        parameters=parameters,
    )


def _create_plot(
    prediction: Series,
    arima_datasets: ArimaDatasets,
    log_label: str = "",
    error_metrics: str = "",
    training_data_plot_extension: int = 200,
) -> Figure:
    fig, ax = pyplot.subplots(figsize=(12, 6))

    prediction.plot(ax=ax, legend=True, linewidth=2, label="Prediction")
    prediction_length = len(arima_datasets.validation_dataset)
    train_plot_length = prediction_length * 2 + training_data_plot_extension
    arima_datasets.train_dataset.tail(train_plot_length).plot(
        ax=ax, legend=True, label="Training"
    )
    arima_datasets.validation_dataset.plot(
        ax=ax, legend=True, label="Actual", linestyle="--"
    )

    ax.set_title(f"{log_label}\n{error_metrics}")
    ax.legend()

    return fig

# Taken from https://github.com/nachi-hebbar/ARIMA-Temperature_Forecasting
import pickle

from pandas import Series
from statsmodels.tsa.statespace.sarimax import SARIMAX

from config import config

from .interface import ArimaOrder


def train_model(
    train_dataset: Series,
    arima_order: ArimaOrder = None,
    should_save_model: bool = True,
) -> SARIMAX:
    """
    Train an ARIMA model using the given train_dataset.
    Parameters:
        train_dataset (Series): The training dataset.
        arima_order contains:
            order (ArimaOrder, optional): The order (p,d,q) of the ARIMA model. Defaults to (1,0,0).
            seasonal_order (ArimaSeasonalOrder, optional): The seasonal order (p,d,q, m) of the ARIMA model.
                Defaults to (0,0,0,0).
        should_save_model (bool, optional): Whether to save the trained model. Defaults to True.
    Returns:
        ARIMA: The trained ARIMA model.
    """
    arima_order = arima_order or ArimaOrder()
    model = SARIMAX(
        train_dataset,
        order=arima_order.order,
        seasonal_order=arima_order.seasonal_order,
    )
    fit_model = model.fit()
    if should_save_model:
        save_model(fit_model)
    return fit_model


def save_model(model, filename: str = config.arima_model_path):
    with open(filename, "wb") as file:
        pickle.dump(model, file)


def load_model(filename: str = config.arima_model_path):
    with open(filename, "rb") as file:
        return pickle.load(file)

from pandas import Series
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .interface import ArimaOrder


def train_model(
    train_dataset: Series,
    arima_order: ArimaOrder = None,
    optimization_method: str = None,
) -> SARIMAX:
    """
    Train an ARIMA model using the given train_dataset.
    Parameters:
        train_dataset (Series): The training dataset.
        arima_order contains:
            order (ArimaOrder, optional): The order (p,d,q) of the ARIMA model. Defaults to (1,0,0).
            seasonal_order (ArimaSeasonalOrder, optional): The seasonal order (p,d,q, m) of the ARIMA model.
                Defaults to (0,0,0,0).
        should_save_model (bool, optional): Whether to save the trained model. Defaults to False.
    Returns:
        ARIMA: The trained ARIMA model.
    """
    arima_order = arima_order or ArimaOrder()
    optimization_method = optimization_method or "lbfgs"
    model = SARIMAX(
        train_dataset,
        order=arima_order.order,
        seasonal_order=arima_order.seasonal_order,
    )
    fit_model = model.fit(method=optimization_method)
    return fit_model

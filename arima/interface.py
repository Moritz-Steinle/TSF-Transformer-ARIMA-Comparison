from pandas import DataFrame


class ArimaDatasets:
    def __init__(self, train_dataset: DataFrame, test_dataset: DataFrame) -> None:
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


class ArimaOrder:
    def __init__(
        self,
        order: tuple[int, int, int],
        seasonal_order: tuple[int, int, int, int] = None,
    ) -> None:
        """
        Contains ARIMA model orders.
        Args:
            order (tuple[int, int, int]): The ARIMA order (p,d,q).
            seasonal_order (tuple[int, int, int, int], optional): The seasonal ARIMA order (p,d,q,m). m is the season length. Defaults to None.
        """
        self.order = order
        self.seasonal_order = seasonal_order

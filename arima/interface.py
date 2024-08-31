from dataclasses import dataclass
from typing import Optional

from pandas import Series


@dataclass
class ArimaDatasets:
    train_dataset: Series
    test_dataset: Series


@dataclass
class ArimaOrder:
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0)


def get_influx_order(resolution: str) -> ArimaOrder:
    if resolution == "24h":
        return ArimaOrder(order=(0, 0, 1), seasonal_order=(0, 0, 1, 6))
    elif resolution == "12h":
        return ArimaOrder(order=(2, 0, 0), seasonal_order=(0, 0, 1, 12))
    elif resolution == "8h":
        return ArimaOrder(order=(1, 0, 1), seasonal_order=(0, 0, 1, 18))
    elif resolution == "6h":
        return ArimaOrder(order=(1, 0, 0), seasonal_order=(2, 0, 0, 24))
    else:
        raise ValueError(f"Resolution {resolution} not supported")


def get_sawtooth_order() -> ArimaOrder:
    return ArimaOrder(order=(0, 0, 0), seasonal_order=(2, 0, 2, 10))

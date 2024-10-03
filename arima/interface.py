from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pandas import Series


@dataclass
class ArimaDatasets:
    train_dataset: Series
    validation_dataset: Series


@dataclass
class ArimaOrder:
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0)


class OptimizationMethod(Enum):
    BFGS = "bfgs"
    L_BFGS = "lbfgs"
    CG = "cg"
    NCG = "ncg"
    POWELL = "powell"


# TODO Add to type hints
class Resolution(Enum):
    H24 = "24h"
    H12 = "12h"
    H8 = "8h"
    H6 = "6h"
    H4 = "4h"
    H3 = "3h"
    H2 = "2h"
    H12_CHAINED = "12h_chained"
    H2_CHAINED_ATDROP_LARGE = "2h-chained_atDrop_large"


def get_influx_order(resolution: str) -> ArimaOrder:
    if resolution == Resolution.H24.value:
        return ArimaOrder(order=(0, 0, 1), seasonal_order=(0, 0, 1, 6))
    elif resolution == Resolution.H12.value:
        return ArimaOrder(order=(2, 0, 0), seasonal_order=(0, 0, 1, 12))
    elif resolution == Resolution.H8.value:
        return ArimaOrder(order=(1, 0, 1), seasonal_order=(0, 0, 1, 18))
    elif resolution == Resolution.H6.value:
        return ArimaOrder(order=(1, 0, 0), seasonal_order=(2, 0, 0, 24))
    elif resolution == Resolution.H4.value:
        return ArimaOrder(order=(1, 1, 0), seasonal_order=(1, 0, 0, 43))
    elif resolution == Resolution.H3.value:
        return ArimaOrder(order=(0, 1, 0), seasonal_order=(0, 0, 0, 56))
    elif resolution == Resolution.H12_CHAINED.value:
        return ArimaOrder(order=(2, 0, 0), seasonal_order=(0, 0, 1, 12))
    else:
        raise ValueError(f"Resolution {resolution} not supported")


def get_sawtooth_order() -> ArimaOrder:
    return ArimaOrder(order=(0, 0, 0), seasonal_order=(2, 0, 2, 10))

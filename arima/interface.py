from dataclasses import dataclass
from typing import Optional

from pandas import DataFrame


@dataclass
class ArimaDatasets:
    train_dataset: DataFrame
    test_dataset: DataFrame


@dataclass
class ArimaOrder:
    order: tuple[int, int, int] = (1, 0, 0)
    seasonal_order: Optional[tuple[int, int, int, int]] = (0, 0, 0, 0)

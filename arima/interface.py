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


class OptimisationMethod(Enum):
    BFGS = "bfgs"
    L_BFGS = "lbfgs"
    CG = "cg"
    NCG = "ncg"
    POWELL = "powell"

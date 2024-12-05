from dataclasses import dataclass


@dataclass
class ErrorMetrics:
    mae: float
    median_ae: float
    smape: float
    rsme: float
    mase: float

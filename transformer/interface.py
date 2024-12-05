from dataclasses import dataclass
from inspect import signature
from typing import Dict, List, Optional, Tuple

from pandas import Series
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader

from config import config


class PurgeNone:
    """
    The PurgeNone class provides methods to filter out attributes
    with None values from an instance's dictionary.
    This is necessay to use unpacking without overwriting default values with None values.
    """

    def none_filtered_dict(self):
        """
        Returns a dictionary of all attributes that are not None.
        """
        return {k: v for k, v in self.__dict__.items() if v is not None}

    def function_none_filtered_dict(self, function: callable) -> dict:
        """
        Returns a dictionary of all attributes that are in the function signature and not None.
        """
        func_params = signature(function).parameters.keys()
        none_filtered_dict = self.none_filtered_dict()
        return {
            k: none_filtered_dict[k] for k in none_filtered_dict if k in func_params
        }


@dataclass
class Dataloaders:
    train_dataloader: DataLoader
    val_dataloader: DataLoader
    training_timeseries: TimeSeriesDataSet
    training_dataset: Series
    validation_dataset: Series


@dataclass
class ModelPath:
    """
    ModelPath stores the path for loading a model
    """

    version: int
    epoch: int
    step: int

    def get_path(self):
        return f"lightning_logs/version_{self.version}/checkpoints/epoch={self.epoch}-step={self.step}.ckpt"


@dataclass
class Hyperparamters(PurgeNone):
    """
    Transformer hyperparameters
    """

    gradient_clip_val: Optional[float] = None
    hidden_continuous_size: Optional[int] = None
    dropout: Optional[float] = None
    attention_head_size: Optional[int] = None
    learning_rate: Optional[float] = None
    accelerator: str = "auto"
    hidden_size: Optional[int] = None


@dataclass
class HyperparameterRanges(PurgeNone):
    """
    HyperparameterRanges defines the ranges for a hyperparameter study.
    """

    gradient_clip_val_range: Optional[Tuple[float, float]] = None
    hidden_size_range: Optional[Tuple[int, int]] = None
    hidden_continuous_size_range: Optional[Tuple[int, int]] = None
    attention_head_size_range: Optional[Tuple[int, int]] = None
    dropout_range: Optional[Tuple[float, float]] = None
    learning_rate_range: Optional[Tuple[float, float]] = None


@dataclass
class DataloaderParameters(PurgeNone):
    """
    DataloaderParameters is a configuration class for setting up parameters for a data loader.
    The configurations are based on the underlying dataset attributes.
    """

    time_idx: str
    target: str
    group_ids: List[str]
    max_prediction_length: int
    batch_size: Optional[int] = 64
    max_encoder_length: Optional[int] = None
    min_encoder_length: Optional[int] = None
    min_prediction_length: Optional[int] = None
    static_categoricals: Optional[List[str]] = None
    static_reals: Optional[List[str]] = None
    time_varying_known_categoricals: Optional[List[str]] = None
    time_varying_known_reals: Optional[List[str]] = None
    time_varying_unknown_categoricals: Optional[List[str]] = None
    time_varying_unknown_reals: Optional[List[str]] = None
    variable_groups: Optional[Dict[str, List[int]]] = None
    add_relative_time_idx: bool = False
    add_target_scales: bool = False
    add_encoder_length: str = "auto"
    target_normalizer: str = "auto"


def get_base_dataloader_parameters(
    max_prediction_length: int,
) -> DataloaderParameters:
    return DataloaderParameters(
        time_idx="time_idx",
        target="value",
        group_ids=["group"],
        min_encoder_length=max_prediction_length * 2,
        max_encoder_length=max_prediction_length * 4,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=["value"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        static_reals=[],
        variable_groups=[],
        static_categoricals=[],
        time_varying_unknown_categoricals=[],
    )

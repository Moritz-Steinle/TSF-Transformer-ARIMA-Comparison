from dataclasses import dataclass
from inspect import signature
from typing import Dict, List, Optional, Tuple

from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from torch.utils.data import DataLoader

from config import config


class PurgeNone:
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
    training_dataset: TimeSeriesDataSet


@dataclass
class DatalaodersAndModel(Dataloaders):
    model: TemporalFusionTransformer


@dataclass
class ModelPath:
    version: int
    epoch: int
    step: int

    def get_path(self):
        return f"lightning_logs/version_{self.version}/checkpoints/epoch={self.epoch}-step={self.step}.ckpt"


@dataclass
class Hyperparamters(PurgeNone):
    gradient_clip_val: Optional[float] = None
    hidden_continuous_size: Optional[int] = None
    dropout: Optional[float] = None
    attention_head_size: Optional[int] = None
    learning_rate: Optional[float] = None
    accelerator: str = "auto"
    max_epochs: int = config.max_epochs
    hidden_size: Optional[int] = None


@dataclass
class HyperparameterRanges(PurgeNone):
    gradient_clip_val_range: Optional[Tuple[float, float]] = None
    hidden_size_range: Optional[Tuple[int, int]] = None
    hidden_continuous_size_range: Optional[Tuple[int, int]] = None
    attention_head_size_range: Optional[Tuple[int, int]] = None
    dropout_range: Optional[Tuple[float, float]] = None
    learning_rate_range: Optional[Tuple[float, float]] = None


@dataclass
class DataloaderParameters(PurgeNone):
    time_idx: str
    target: str
    group_ids: List[str]
    max_prediction_length: int = config.max_prediction_length
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


def get_stallion_dataset_parameters() -> DataloaderParameters:
    special_days = [
        "easter_day",
        "good_friday",
        "new_year",
        "christmas",
        "labor_day",
        "independence_day",
        "revolution_day_memorial",
        "regional_games",
        "fifa_u_17_world_cup",
        "football_gold_cup",
        "beer_capital",
        "music_fest",
    ]
    max_prediction_length = 6
    return DataloaderParameters(
        time_idx="time_idx",
        target="volume",
        group_ids=["agency", "sku"],
        min_encoder_length=max_prediction_length * 2,
        max_encoder_length=max_prediction_length * 4,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["agency", "sku"],
        static_reals=["avg_population_2017", "avg_yearly_household_income_2017"],
        time_varying_known_categoricals=["special_days", "month"],
        variable_groups={
            "special_days": special_days
        },  # group of categorical variables can be treated as one variable
        time_varying_known_reals=["time_idx", "price_regular", "discount_in_percent"],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[
            "volume",
            "log_volume",
            "industry_volume",
            "soda_volume",
            "avg_max_temp",
            "avg_volume_by_agency",
            "avg_volume_by_sku",
        ],
        target_normalizer=GroupNormalizer(
            groups=["agency", "sku"], transformation="softplus"
        ),  # use softplus and normalize by group
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


def get_influx_dataloader_parameters() -> DataloaderParameters:
    max_prediction_length = 150
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


def get_sawtooth_hyperparameters() -> Hyperparamters:
    return Hyperparamters(
        gradient_clip_val=0.5286525368230415,
        dropout=0.28145444638341954,
        hidden_continuous_size=18,
        attention_head_size=3,
        learning_rate=0.572695014452453,
    )


def get_influx_hyperparameters() -> Hyperparamters:
    return Hyperparamters(
        gradient_clip_val=0.09050490030726796,
        dropout=0.22288661702971777,
        hidden_continuous_size=12,
        attention_head_size=2,
        learning_rate=0.6336776189720053,
    )


def get_stallion_hyperparameters() -> Hyperparamters:
    return Hyperparamters(
        gradient_clip_val=0.09050490030726796,
        dropout=0.22288661702971777,
        hidden_continuous_size=12,
        attention_head_size=2,
        learning_rate=0.6336776189720053,
    )

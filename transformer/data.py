import os
from datetime import datetime

import pandas as pd
from optuna import Study
from optuna.trial import FrozenTrial
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

from config import config
from transformer.interface import (
    DataloaderParameters,
    Dataloaders,
    HyperparameterRanges,
    Hyperparamters,
)


def create_dataloaders(
    dataset: DataFrame,
    dataloader_parameters: DataloaderParameters,
) -> Dataloaders:
    training_cutoff = (
        dataset[dataloader_parameters.time_idx].max()
        - dataloader_parameters.max_prediction_length
    )
    _parameters = dataloader_parameters.function_none_filtered_dict(
        function=TimeSeriesDataSet
    )
    training_dataset = TimeSeriesDataSet(
        dataset[lambda x: x.time_idx <= training_cutoff],
        **_parameters,
    )
    validation = TimeSeriesDataSet.from_dataset(
        training_dataset, dataset, predict=True, stop_randomization=True
    )
    train_dataloader = training_dataset.to_dataloader(
        train=True, batch_size=dataloader_parameters.batch_size, num_workers=11
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=dataloader_parameters.batch_size * 10, num_workers=11
    )
    return Dataloaders(train_dataloader, val_dataloader, training_dataset)


def run_hyperparameter_study(
    train_dataloader,
    val_dataloader,
    amount_trials: int = 50,
    hyperparameter_ranges: HyperparameterRanges = HyperparameterRanges(),
    study_filename: str = datetime.now(),
) -> Hyperparamters:
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=amount_trials,
        **hyperparameter_ranges.none_filtered_dict(),
    )
    save_hyperparameter_study(study, study_filename)
    return Hyperparamters(**study.best_trial.params)


def save_hyperparameter_study(study: Study, filename: str = datetime.now()) -> None:
    trials_list = study.trials
    all_trials_df = pd.concat(
        [trial_to_dataframe(trial) for trial in trials_list], ignore_index=True
    )
    all_trials_df = all_trials_df.sort_values(by="value")
    full_path = os.path.join(config.hyperparameter_study_path, f"{filename}.json")
    all_trials_df.to_json(full_path, orient="records", lines=True)


def trial_to_dataframe(trial: FrozenTrial) -> DataFrame:
    data = {
        "value": trial.values[0] if trial.values else None,
        "trial_number": trial.number,
        "duration": (trial.datetime_complete - trial.datetime_start).total_seconds()
        if trial.datetime_complete
        else None,
    }
    data.update(trial.params)
    df = DataFrame([data])
    return df

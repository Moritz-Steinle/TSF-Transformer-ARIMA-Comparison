import pandas as pd
from optuna import Study
from optuna.trial import FrozenTrial
from pandas import DataFrame
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

from config import config
from log.log import get_path_with_timestamp
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
    """
    Create dataloaders for training and validation datasets.
    Args:
        dataset (DataFrame): The input dataset.
        dataloader_parameters (DataloaderParameters): Parameters for creating dataloaders.
    Returns:
        Dataloaders: Train and validation dataloaders.
    """
    training_cutoff = (
        dataset[dataloader_parameters.time_idx].max()
        - dataloader_parameters.max_prediction_length
    )
    training_dataset = dataset[lambda x: x.time_idx <= training_cutoff]
    validation_dataset = dataset[lambda x: x.time_idx > training_cutoff]
    if training_cutoff <= 0:
        raise ValueError(
            f"Max prediction length {dataloader_parameters.max_prediction_length} is too large for dataset"
        )
    _parameters = dataloader_parameters.function_none_filtered_dict(
        function=TimeSeriesDataSet
    )
    training_timeseries = TimeSeriesDataSet(
        training_dataset,
        **_parameters,
    )
    train_dataloader = training_timeseries.to_dataloader(
        train=True, batch_size=dataloader_parameters.batch_size, num_workers=11
    )
    validation = TimeSeriesDataSet.from_dataset(
        training_timeseries, dataset, predict=True, stop_randomization=True
    )
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=dataloader_parameters.batch_size * 10, num_workers=11
    )
    return Dataloaders(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        training_timeseries=training_timeseries,
        training_dataset=training_dataset,
        validation_dataset=validation_dataset,
    )


def run_hyperparameter_study(
    train_dataloader,
    val_dataloader,
    hyperparameters_study_trials: int = 50,
    hyperparameter_ranges: HyperparameterRanges = HyperparameterRanges(),
) -> Hyperparamters:
    """
    Conducts a hyperparameter study using Optuna to optimize model performance.
    Args:
        train_dataloader: DataLoader for the training dataset.
        val_dataloader: DataLoader for the validation dataset.
        hyperparameters_study_trials (int, optional): Number of trials for the hyperparameter study. Defaults to 50.
        hyperparameter_ranges (HyperparameterRanges, optional): Ranges for the hyperparameters to be optimized.
            Defaults to HyperparameterRanges().
    Returns:
        Hyperparamters: The best hyperparameters found during the study.
    """
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=hyperparameters_study_trials,
        **hyperparameter_ranges.none_filtered_dict(),
    )
    save_hyperparameter_study(study)
    return Hyperparamters(**study.best_trial.params)


def save_hyperparameter_study(study: Study) -> None:
    """
    Save the hyperparameter study results to a JSON file.
    Parameters:
        study (Study): The hyperparameter study object.
    Returns:
        None
    """
    study_dataframe = study.trials_dataframe()
    study_dataframe = study_dataframe.sort_values(by="value")
    full_path = get_path_with_timestamp(config.hyperparameter_study_path, "json")
    study_dataframe.to_json(full_path, orient="records", lines=True)

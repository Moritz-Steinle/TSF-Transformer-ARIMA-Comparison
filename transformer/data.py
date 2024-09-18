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
from util import get_path_with_timestamp


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
    if training_cutoff <= 0:
        raise ValueError(
            f"Max prediction length {dataloader_parameters.max_prediction_length} is too large for dataset"
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
    hyperparameters_study_trials: int = 50,
    hyperparameter_ranges: HyperparameterRanges = HyperparameterRanges(),
) -> Hyperparamters:
    """
    Runs a hyperparameter study using Optuna to optimize the model's hyperparameters.
    Args:
        train_dataloader: The dataloader for the training data.
        val_dataloader: The dataloader for the validation data.
        amount_trials (int): The number of trials to run for hyperparameter optimization. Default is 50.
        hyperparameter_ranges (HyperparameterRanges): The ranges to test the hyperparameters in. Default is an empty HyperparameterRanges object.
        study_filename (str): The filename to save the hyperparameter study. Default is the current datetime.
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
    trials_list = study.trials
    all_trials_df = pd.concat(
        [_trial_to_dataframe(trial) for trial in trials_list], ignore_index=True
    )
    all_trials_df = all_trials_df.sort_values(by="value")
    full_path = get_path_with_timestamp(config.hyperparameter_study_path, "json")
    all_trials_df.to_json(full_path, orient="records", lines=True)


def _trial_to_dataframe(trial: FrozenTrial) -> DataFrame:
    """
    Helper function to filter relevant trial object data into a pandas DataFrame.
    Parameters:
        trial (FrozenTrial): The trial object to convert.
    Returns:
        DataFrame: A pandas DataFrame containing relevant trial data.
    """

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

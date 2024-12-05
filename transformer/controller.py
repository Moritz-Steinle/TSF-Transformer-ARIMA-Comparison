import time

from pandas import DataFrame

import transformer.data
import transformer.evaluation
import transformer.model
from transformer.interface import (
    DataloaderParameters,
    Dataloaders,
    HyperparameterRanges,
    Hyperparamters,
)


def train_and_evaluate_transformer(
    dataset: DataFrame,
    max_training_epochs: int,
    dataloader_parameters: DataloaderParameters = None,
    hyperparameters: Hyperparamters = None,
    hyperparameters_study_trials: int = 0,
    log_label: str = "",
    fast_dev_run: bool = False,
):
    """
    Train and evaluate a transformer model on the given dataset.

    Args:
        dataset (DataFrame): The dataset to be used for training and evaluation.
        max_training_epochs (int): The maximum number of training epochs.
        dataloader_parameters (DataloaderParameters, optional): Parameters for the dataloader. Defaults to None.
        hyperparameters (Hyperparamters, optional): Hyperparameters for the model. Defaults to None.
        hyperparameters_study_trials (int, optional): Number of trials for hyperparameter study. Defaults to 0.
        log_label (str, optional): Label for logging. Defaults to "".
        fast_dev_run (bool, optional): If True, runs a fast development run. Defaults to False.

    Returns:
        None
    """
    dataloaders = transformer.data.create_dataloaders(
        dataset, dataloader_parameters=dataloader_parameters
    )
    hyperparameters, hyperparameters_study_runtime = run_hyperparameter_study(
        dataloaders=dataloaders,
        hyperparameters=hyperparameters,
        hyperparameters_study_trials=hyperparameters_study_trials,
    )
    start_time = time.time()
    result = transformer.model.train_model(
        dataloaders=dataloaders,
        max_epochs=max_training_epochs,
        hyperparameters=hyperparameters,
        fast_dev_run=fast_dev_run,
    )
    training_runtime = time.time() - start_time
    if fast_dev_run:
        return
    prediction = transformer.evaluation.make_prediction(
        result.model, result.dataloaders.val_dataloader
    )
    transformer.evaluation.log(
        prediction=prediction,
        dataloaders=dataloaders,
        max_epochs=max_training_epochs,
        hyperparameters=hyperparameters,
        log_label=log_label,
        training_runtime=training_runtime,
        hyperparameters_study_runtime=hyperparameters_study_runtime,
    )


def run_hyperparameter_study(
    dataloaders: Dataloaders,
    hyperparameters: Hyperparamters = None,
    hyperparameters_study_trials: int = 1,
    hyperparameter_ranges: HyperparameterRanges = None,
) -> tuple[Hyperparamters, float]:
    """
    Either returns given hyperparameters or runs a hyperparameter study.
    Args:
        dataloaders (Dataloaders): The dataloaders containing the train and validation dataloaders.
        hyperparameters (Hyperparamters, optional): The hyperparameters to use instead of ding a study.
            Defaults to None.
        hyperparameters_study_trials (int, optional): The number of trials to run in the hyperparameter study.
            Defaults to 1.
        hyperparameter_ranges (HyperparameterRanges, optional):
            In what range to test the hyperparameters. Can make search more efficient. Defaults to an empty class.
    Returns:
        tuple[Hyperparamters, float]: The hyperparameters and the runtime of the hyperparameter study.
    """
    if hyperparameters is not None:
        return hyperparameters, None
    elif hyperparameters_study_trials > 0:
        start_time = time.time()
        hyperparameter_ranges = hyperparameter_ranges or HyperparameterRanges()
        hyperparameters = transformer.data.run_hyperparameter_study(
            dataloaders.train_dataloader,
            dataloaders.val_dataloader,
            hyperparameters_study_trials=hyperparameters_study_trials,
            hyperparameter_ranges=hyperparameter_ranges,
        )
        hyperparameters_study_runtime = time.time() - start_time
        return hyperparameters, hyperparameters_study_runtime
    else:
        raise ValueError(
            "Either provide hyperparameters or set hyperparameters_study_trials to a value greater than 0."
        )

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
    max_epochs: int,
    dataloader_parameters: DataloaderParameters = None,
    hyperparameters: Hyperparamters = None,
    hyperparameters_study_trials: int = 0,
    fast_dev_run: bool = False,
):
    """
    Trains and evaluates a transformer model.
    Args:
        dataset (DataFrame): The dataset to train and evaluate the transformer model on.
        max_epochs (int): The maximum number of epochs to train the model.
        dataloader_parameters (DataloaderParameters, optional):
            Optimizing parameters specific to the data like learning rate etc.  Defaults to None.
        hyperparameters (Hyperparamters, optional): The hyperparameters for training the model. Defaults to None.
        should_run_hyperparameter_study (bool, optional): Whether to run a hyperparameter study. Defaults to False.
        fast_dev_run (bool, optional):
            Whether to run a fast development run. This runs only 1 training epoch and yields no result. Defaults to False.
    Raises:
        ValueError: If neither hyperparameters are provided nor should_run_hyperparameter_study is set to True.
    Returns:
        None: If fast_dev_run is True.
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
        max_epochs=max_epochs,
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
        model=result.model,
        prediction=prediction,
        dataloaders=dataloaders,
        hyperparameters=hyperparameters,
        log_label="Transformer",
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

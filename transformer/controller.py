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
    dataloader_parameters: DataloaderParameters = None,
    hyperparameters: Hyperparamters = None,
    should_run_hyperparameter_study: bool = False,
    fast_dev_run: bool = False,
):
    """
    Trains and evaluates a transformer model.
    Args:
        dataset (DataFrame): The dataset to train and evaluate the transformer model on.
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
    if should_run_hyperparameter_study:
        _hyperparameters = run_hyperparameter_study(dataloaders)
    elif hyperparameters is not None:
        _hyperparameters = hyperparameters
    else:
        raise ValueError(
            "Either provide hyperparameters or set should_run_hyperparameter_study to True"
        )
    result = transformer.model.train_model(
        dataloaders=dataloaders,
        hyperparameters=_hyperparameters,
        fast_dev_run=fast_dev_run,
    )
    if fast_dev_run:
        return
    prediction = transformer.evaluation.make_prediction(
        result.model, result.dataloaders.val_dataloader
    )
    transformer.evaluation.log(
        model=result.model,
        prediction=prediction,
        hyperparameters=_hyperparameters,
        log_label="Transformer",
        training_runtime=-1,
    )


def run_hyperparameter_study(
    dataloaders: Dataloaders,
    hyperparameter_ranges: HyperparameterRanges = None,
) -> Hyperparamters:
    """
    Run a hyperparameter study using the given dataloaders and hyperparameter ranges.
    Args:
        dataloaders (Dataloaders): The dataloaders containing the train and validation dataloaders.
        hyperparameter_ranges (HyperparameterRanges, optional):
            In what range to test the hyperparameters. Can make search more efficient. Defaults to an empty class.
    Returns:
        Hyperparamters: The hyperparameters obtained from the study.
    """
    if hyperparameter_ranges is None:
        hyperparameter_ranges = HyperparameterRanges()
    return transformer.data.run_hyperparameter_study(
        dataloaders.train_dataloader,
        dataloaders.val_dataloader,
        amount_trials=1,
        hyperparameter_ranges=hyperparameter_ranges,
    )

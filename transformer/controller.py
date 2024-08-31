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


def run_hyperparameter_study(
    dataloaders: Dataloaders,
    hyperparameter_rangers: HyperparameterRanges = HyperparameterRanges(),
) -> Hyperparamters:
    return transformer.data.run_hyperparameter_study(
        dataloaders.train_dataloader,
        dataloaders.val_dataloader,
        amount_trials=1,
        hyperparameter_ranges=hyperparameter_rangers,
    )


def train_and_evaluate_transformer(
    dataset: DataFrame,
    dataloader_parameters: DataloaderParameters = None,
    hyperparameters: Hyperparamters = None,
    should_run_hyperparameter_study: bool = False,
    fast_dev_run: bool = False,
):
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
    transformer.evaluation.make_prediction(
        result.model, result.dataloaders.val_dataloader
    )
    print(hyperparameters)

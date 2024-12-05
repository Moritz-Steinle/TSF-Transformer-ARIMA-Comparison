from pandas import Series
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.base_model import Prediction

from log.log import log_prediction

from .interface import Dataloaders, Hyperparamters


def make_prediction(model: TemporalFusionTransformer, val_dataloader) -> Prediction:
    """
    Makes a prediction using the trained model
    Args:
        model (TemporalFusionTransformer): The trained model
        val_dataloader: The validation dataloader
    Returns:
        Prediction: The prediction results
    """
    return model.predict(
        val_dataloader,
        mode="prediction",
    )


def log(
    prediction: Prediction,
    dataloaders: Dataloaders,
    max_epochs: int,
    hyperparameters: Hyperparamters,
    training_runtime: float,
    hyperparameters_study_runtime: float = None,
    log_label: str = None,
) -> None:
    """
    Formats and logs training and prediction data
    Args:
        prediction (Prediction): The prediction results from the model.
        dataloaders (Dataloaders): The dataloaders containing training and validation datasets.
        max_epochs (int): The maximum number of epochs used during training.
        hyperparameters (Hyperparamters): The hyperparameters used for training the model.
        training_runtime (float): The total runtime of the training process in seconds.
        hyperparameters_study_runtime (float, optional): The runtime of the hyperparameter study in seconds.
            Defaults to None.
        log_label (str, optional): An optional label for the log entry. Defaults to None.
    Returns:
        None
    """
    runtimes = f"Training: {training_runtime:.2f} seconds"
    if hyperparameters_study_runtime is not None:
        runtimes += (
            f" , hyperparameter study: {hyperparameters_study_runtime:.2f} seconds)"
        )
    parameters = f"Epochs: {max_epochs}, hyperparameters: {hyperparameters}"
    validation_dataset = dataloaders.validation_dataset["value"]
    prediction_values = _prediction_to_list(prediction, validation_dataset)
    log_prediction(
        model="Transformer",
        prediction=prediction_values,
        training_dataset=dataloaders.training_dataset["value"],
        validation_dataset=validation_dataset,
        label=log_label,
        runtimes=runtimes,
        parameters=parameters,
    )


def _prediction_to_list(prediction: Prediction, validation_dataset: Series) -> Series:
    """
    Exctracts the absolute prediction values from the prediction object to a Series
    Args:
        prediction (Prediction): The prediction object containing the prediction data.
        validation_dataset (Series): The validation dataset whose index will be used for the resulting Series.

    Returns:
        Series: A pandas Series containing the prediction data with the same index as the validation dataset.
    """
    #
    prediction_series = Series(prediction.data.squeeze().tolist())
    prediction_series.index = validation_dataset.index
    return prediction_series

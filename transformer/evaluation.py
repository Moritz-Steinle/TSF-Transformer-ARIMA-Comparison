import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pandas import Series
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.base_model import Prediction

from log.log import log_prediction

from .interface import Dataloaders, Hyperparamters


def make_prediction(model: TemporalFusionTransformer, val_dataloader) -> Prediction:
    return model.predict(
        val_dataloader,
        return_x=True,
        mode="raw",
    )


def log(
    model: TemporalFusionTransformer,
    prediction: Prediction,
    dataloaders: Dataloaders,
    max_epochs: int,
    hyperparameters: Hyperparamters,
    training_runtime: float,
    hyperparameters_study_runtime: float = None,
    log_label: str = None,
) -> None:
    runtimes = f"Training: {training_runtime:.2f} seconds"
    if hyperparameters_study_runtime is not None:
        runtimes += (
            f" , hyperparameter study: {hyperparameters_study_runtime:.2f} seconds)"
        )
    parameters = f"Epochs: {max_epochs}, hyperparameters: {hyperparameters}"
    prediction_values = _prediction_to_list(prediction)
    plot = _create_plot(model, prediction)
    log_prediction(
        model="Transformer",
        prediction=prediction_values,
        plot=plot,
        training_dataset=dataloaders.training_dataset["value"],
        validation_dataset=dataloaders.validation_dataset["value"],
        label=log_label,
        runtimes=runtimes,
        parameters=parameters,
    )


def _create_plot(model: TemporalFusionTransformer, predictions) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    network_input = predictions.x
    network_output = predictions.output
    model.plot_prediction(network_input, network_output, idx=0, ax=ax)
    plt.tight_layout()
    return fig


def _prediction_to_list(prediction: Prediction) -> Series:
    tensor_data = prediction.output.prediction
    return Series(tensor_data.squeeze().tolist())

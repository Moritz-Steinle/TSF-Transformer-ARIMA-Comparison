from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.base_model import Prediction

from util import log_prediction

from .interface import Hyperparamters


def make_prediction(model: TemporalFusionTransformer, val_dataloader) -> Prediction:
    return model.predict(
        val_dataloader,
        return_x=True,
        mode="raw",
    )


def log(
    model: TemporalFusionTransformer,
    prediction: Prediction,
    hyperparameters: Hyperparamters,
    training_runtime: float,
    hyperparameter_study_runtime: float = None,
    log_label: str = None,
) -> None:
    runtimes = f"Training: {training_runtime:.2f} seconds"
    if hyperparameter_study_runtime is not None:
        runtimes += (
            f" , hyperparameter study: {hyperparameter_study_runtime:.2f} seconds)"
        )
    parameters = f"Hyperparameters: {hyperparameters}"
    plot = _create_plot(model, prediction)
    prediction_string = str(prediction)
    log_prediction(
        model="Transformer",
        prediction=prediction_string,
        plot=plot,
        label=log_label,
        runtimes=runtimes,
        parameters=parameters,
    )


def _create_plot(model: TemporalFusionTransformer, predictions):
    network_input = predictions.x
    network_output = predictions.output
    return model.plot_prediction(
        network_input, network_output, idx=0, add_loss_to_title=True
    )

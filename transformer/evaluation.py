from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics.point import RMSE, SMAPE
from pytorch_forecasting.models.base_model import Prediction
from torch.utils.data import DataLoader

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
    target_col: str = "value",
    hyperparameters_study_runtime: float = None,
    log_label: str = None,
) -> None:
    runtimes = f"Training: {training_runtime:.2f} seconds"
    if hyperparameters_study_runtime is not None:
        runtimes += (
            f" , hyperparameter study: {hyperparameters_study_runtime:.2f} seconds)"
        )
    parameters = f"Epochs: {max_epochs}, hyperparameters: {hyperparameters}"
    plot = _create_plot(model, prediction)
    prediction_values = _prediction_to_list(prediction)
    error_metrics = _calculate_error_metrics(model, dataloaders.val_dataloader)
    log_prediction(
        model="Transformer",
        prediction=prediction_values,
        plot=plot,
        error_metrics=error_metrics,
        # TODO fix dataset length
        label=log_label,
        runtimes=runtimes,
        parameters=parameters,
    )


def _calculate_error_metrics(
    model: TemporalFusionTransformer,
    val_dataloader: DataLoader,
) -> str:
    prediction = model.predict(val_dataloader, return_x=True, return_y=True)
    rsme = RMSE()(prediction.output, prediction.y)
    smape = SMAPE()(prediction.output, prediction.y)
    return f"RMSE: {rsme}, SMAPE: {smape}"


def _create_plot(model: TemporalFusionTransformer, predictions):
    network_input = predictions.x
    network_output = predictions.output
    return model.plot_prediction(
        network_input, network_output, idx=0, add_loss_to_title=True
    )


def _prediction_to_list(prediction: Prediction) -> list:
    tensor_data = prediction.output.prediction
    return tensor_data.squeeze().tolist()

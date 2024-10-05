import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting.metrics.point import MAE, RMSE, SMAPE
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
    hyperparameters_study_runtime: float = None,
    log_label: str = None,
) -> None:
    runtimes = f"Training: {training_runtime:.2f} seconds"
    if hyperparameters_study_runtime is not None:
        runtimes += (
            f" , hyperparameter study: {hyperparameters_study_runtime:.2f} seconds)"
        )
    parameters = f"Epochs: {max_epochs}, hyperparameters: {hyperparameters}"
    error_metrics = _calculate_error_metrics(model, dataloaders.val_dataloader)
    plot_title = f"{log_label}\n{error_metrics}"
    plot = _create_plot(model, prediction, plot_title)
    prediction_values = _prediction_to_list(prediction)
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
    mae = MAE()(prediction.output, prediction.y)
    smape = SMAPE()(prediction.output, prediction.y)
    mase = _calculate_mase_error(val_dataloader=val_dataloader, model_mae=mae)
    return f"MAE: {mae}, SMAPE: {smape}, MASE: {mase}"


def _calculate_mase_error(
    val_dataloader: DataLoader,
    model_mae: float,
) -> float:
    naive_prediction = Baseline().predict(val_dataloader, return_y=True)
    naive_mae = MAE()(naive_prediction.output, naive_prediction.y)
    return model_mae / naive_mae


def _create_plot(
    model: TemporalFusionTransformer, predictions, title: str = ""
) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 6))
    network_input = predictions.x
    network_output = predictions.output
    model.plot_prediction(network_input, network_output, idx=0, ax=ax)
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def _prediction_to_list(prediction: Prediction) -> list:
    tensor_data = prediction.output.prediction
    return tensor_data.squeeze().tolist()

import itertools

from pandas import DataFrame
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.models.base_model import Prediction
from sklearn.metrics import root_mean_squared_error
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error

from util import log_prediction

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
    plot = _create_plot(model, prediction)
    prediction_string = "".join(
        list(itertools.chain.from_iterable(prediction.output.prediction))
    )

    # error_metrics = _calculate_error_metrics(
    #     dataloaders.training_dataset, dataloaders.val_dataloader.dataset, prediction
    # )
    log_prediction(
        model="Transformer",
        prediction=prediction_string,
        # error_metrics=error_metrics,
        length_test_dataset=len(dataloaders.val_dataloader.dataset),
        length_train_dataset=len(dataloaders.train_dataloader.dataset),
        plot=plot,
        label=log_label,
        runtimes=runtimes,
        parameters=parameters,
    )


def _calculate_error_metrics(
    train_dataset: DataFrame,
    test_dataset: DataFrame,
    prediction,
    season_length: int = 1,
) -> str:
    rsme = root_mean_squared_error(test_dataset, prediction)
    mase = mean_absolute_scaled_error(
        y_true=test_dataset,
        y_pred=prediction,
        y_train=train_dataset,
        season_length=season_length,
    )
    return f"RSME: {rsme}, MASE: {mase}"


def _create_plot(model: TemporalFusionTransformer, predictions):
    network_input = predictions.x
    network_output = predictions.output
    return model.plot_prediction(
        network_input, network_output, idx=0, add_loss_to_title=True
    )

import os
import warnings

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics.point import MAE

from config import project_root_path

from .interface import Dataloaders, Hyperparamters, ModelPath

warnings.filterwarnings("ignore", category=Warning, module="sklearn")


def train_model(
    dataloaders: Dataloaders,
    max_epochs: int,
    hyperparameters: Hyperparamters = None,
    fast_dev_run: bool = False,
) -> LightningModule | None:
    """
    Trains a TemporalFusionTransformer model with Lightning-Trainer using the provided dataloaders and hyperparameters.
    Args:
        dataloaders (Dataloaders): Contains the training and validation dataloaders.
        max_epochs (int): The maximum number of epochs to train the model.
        hyperparameters (Hyperparamters, optional): Contains the hyperparameters for the model.
            Defaults to None.
        fast_dev_run (bool, optional): If True, runs a single batch of training and validation for quick debugging.
            Defaults to False.
    Returns:
        LightningModule | None: The trained TemporalFusionTransformer model, or None if fast_dev_run is True.
    """
    if hyperparameters is None:
        hyperparameters = Hyperparamters()
    trainer = create_trainer(
        max_epochs=max_epochs,
        hyperparameters=hyperparameters,
        fast_dev_run=fast_dev_run,
    )
    _hyperparameters = hyperparameters.function_none_filtered_dict(
        function=TemporalFusionTransformer.from_dataset
    )
    temporal_fusion_transformer = TemporalFusionTransformer.from_dataset(
        dataset=dataloaders.training_timeseries,
        **_hyperparameters,
        loss=MAE(),
        log_interval=10,
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
        output_size=1,
    )
    trainer.fit(
        temporal_fusion_transformer,
        dataloaders.train_dataloader,
        dataloaders.val_dataloader,
    )
    if fast_dev_run:
        return
    return temporal_fusion_transformer


def create_trainer(
    max_epochs: int,
    hyperparameters: Hyperparamters = None,
    fast_dev_run: bool = False,
) -> Trainer:
    """
    Creates a PyTorch Lightning Trainer from specified configurations.
    Args:
        max_epochs (int): The maximum number of epochs to train the model.
        hyperparameters (Hyperparamters, optional): An instance of Hyperparamters containing training configurations.
            Defaults to None in which case Trainer defaults are used.
        fast_dev_run (bool, optional): If True, runs a single batch of training and validation to quickly check
            for errors. Defaults to False.
    Returns:
        Trainer: A configured PyTorch Lightning Trainer instance.
    """
    if hyperparameters is None:
        hyperparameters = Hyperparamters()
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    _hyperparameters = hyperparameters.function_none_filtered_dict(function=Trainer)
    return Trainer(
        **_hyperparameters,
        max_epochs=max_epochs,
        enable_model_summary=True,
        limit_train_batches=50,
        callbacks=[early_stop_callback],
        fast_dev_run=fast_dev_run,
    )


def load_model(model_path: ModelPath) -> TemporalFusionTransformer:
    """
    Loads a TemporalFusionTransformer model from a specified checkpoint path.
    Args:
        model_path (ModelPath): An instance of ModelPath that provides the path to the model checkpoint.
    Returns:
        TemporalFusionTransformer: The loaded TemporalFusionTransformer model.
    """

    path = model_path.get_path()
    lightning_logs_path = os.path.join(project_root_path, path)
    return TemporalFusionTransformer.load_from_checkpoint(lightning_logs_path)

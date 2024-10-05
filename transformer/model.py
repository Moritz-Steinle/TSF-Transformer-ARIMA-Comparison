import os
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics.point import RMSE

from config import project_root_path

from .interface import DatalaodersAndModel, Dataloaders, Hyperparamters, ModelPath

warnings.filterwarnings("ignore", category=Warning, module="sklearn")


def train_model(
    dataloaders: Dataloaders,
    max_epochs: int,
    hyperparameters: Hyperparamters = Hyperparamters(),
    fast_dev_run: bool = False,
) -> DatalaodersAndModel | None:
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
        loss=RMSE(),
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
    return DatalaodersAndModel(
        dataloaders=dataloaders,
        model=temporal_fusion_transformer,
    )


def create_trainer(
    max_epochs: int,
    hyperparameters: Hyperparamters = Hyperparamters(),
    fast_dev_run: bool = False,
) -> pl.Trainer:
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("")  # logging results to a tensorboard
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    _hyperparameters = hyperparameters.function_none_filtered_dict(function=pl.Trainer)
    return pl.Trainer(
        **_hyperparameters,
        max_epochs=max_epochs,
        enable_model_summary=True,
        limit_train_batches=50,
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
        fast_dev_run=fast_dev_run,
    )


def load_model(model_path: ModelPath = None) -> TemporalFusionTransformer:
    path = model_path.get_path()
    lightning_logs_path = os.path.join(project_root_path, path)
    return TemporalFusionTransformer.load_from_checkpoint(lightning_logs_path)

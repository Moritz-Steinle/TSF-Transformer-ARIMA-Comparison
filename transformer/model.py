# From https://pytorch-forecasting.readthedocs.io/en/stable/tutorials/stallion.html
import os
import warnings

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

from config import config, project_root_path

from .interface import DatalaodersAndModel, Dataloaders, Hyperparamters, ModelPath

# TODO: fix UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
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
    tft = TemporalFusionTransformer.from_dataset(
        dataset=dataloaders.training_dataset,
        **_hyperparameters,
        loss=QuantileLoss(),
        log_interval=10,
        optimizer="Ranger",
        reduce_on_plateau_patience=4,
        # output_size=1,  # TODO No classification problem so i want just one output, not probability of each class
    )
    trainer.fit(
        tft,
        dataloaders.train_dataloader,
        dataloaders.val_dataloader,
    )
    if fast_dev_run:
        return
    # TODO use trainer instead of saving and loading model
    best_model_path = trainer.checkpoint_callback.best_model_path
    save_best_model_path(best_model_path)
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    #
    return DatalaodersAndModel(
        dataloaders=dataloaders,
        model=best_model,
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


def save_best_model_path(best_model_path: str) -> None:
    path = os.path.join(project_root_path, config.best_model_path)
    with open(path, "w") as f:
        f.write(best_model_path)


def load_model(model_path: ModelPath = None) -> TemporalFusionTransformer:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(current_dir, "best_model_path.txt")
    if model_path:
        path = model_path.get_path()
    else:
        with open(default_path) as f:
            path = f.readline()
    lightning_logs_path = os.path.join(project_root_path, path)
    return TemporalFusionTransformer.load_from_checkpoint(lightning_logs_path)

#!/usr/bin/env python
"""
Train Amorphous Flow Matching Model

使用方法:
    # 使用 NequIP (默认)
    python diffcsp/train_amorphous.py
    
    # 使用 EGNN
    python diffcsp/train_amorphous.py model=amorphous_flow_egnn
    
    # 使用 SchNet
    python diffcsp/train_amorphous.py model=amorphous_flow_schnet
    
    # 自定义实验名称
    python diffcsp/train_amorphous.py expname=my-experiment
    
    # 修改训练参数
    python diffcsp/train_amorphous.py model.learning_rate=1e-3 data.datamodule.batch_size.train=64
    
    # 禁用 wandb (离线模式)
    python diffcsp/train_amorphous.py logging.wandb.mode=offline
    
    # Debug 模式
    python diffcsp/train_amorphous.py train.pl_trainer.fast_dev_run=true
"""

import sys
sys.path.append('.')

from pathlib import Path
from typing import List

import hydra
import numpy as np
import torch
import lightning as pl
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from lightning import seed_everything, Callback
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
    RichProgressBar,
)
from lightning.pytorch.profilers import SimpleProfiler as Profiler
from lightning.pytorch.loggers import WandbLogger

from diffcsp.common.utils import log_hyperparameters, PROJECT_ROOT


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    """Build training callbacks."""
    callbacks: List[Callback] = []

    # Learning rate monitor
    if "lr_monitor" in cfg.logging:
        hydra.utils.log.info("Adding callback <LearningRateMonitor>")
        callbacks.append(
            LearningRateMonitor(
                logging_interval=cfg.logging.lr_monitor.logging_interval,
                log_momentum=cfg.logging.lr_monitor.log_momentum,
            )
        )

    # Early stopping
    if "early_stopping" in cfg.train:
        hydra.utils.log.info("Adding callback <EarlyStopping>")
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                patience=cfg.train.early_stopping.patience,
                verbose=cfg.train.early_stopping.verbose,
                min_delta=cfg.train.early_stopping.get('min_delta', 0.0),
            )
        )

    # Model checkpoints
    if HydraConfig.initialized():
        ckpt_dir = str(Path(HydraConfig.get().run.dir).absolute())
    else:
        ckpt_dir = None
        
    if "model_checkpoints" in cfg.train:
        hydra.utils.log.info("Adding callback <ModelCheckpoint>")
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dir,
                monitor=cfg.train.monitor_metric,
                mode=cfg.train.monitor_metric_mode,
                save_top_k=cfg.train.model_checkpoints.save_top_k,
                verbose=cfg.train.model_checkpoints.verbose,
                save_last=cfg.train.model_checkpoints.save_last,
                filename=cfg.train.model_checkpoints.get('filename', None),
                auto_insert_metric_name=cfg.train.model_checkpoints.get('auto_insert_metric_name', True),
            )
        )

    # Progress bar
    callbacks.append(
        TQDMProgressBar(
            refresh_rate=cfg.logging.progress_bar_refresh_rate,
        )
    )

    return callbacks


def get_wandb_logger(cfg, save_dir):
    """Initialize Weights & Biases logger."""
    wandb_logger = None
    if "wandb" in cfg.logging:
        hydra.utils.log.info("Instantiating <WandbLogger>")
        wandb_config = OmegaConf.to_container(cfg.logging.wandb, resolve=True)
        
        # Remove tags from config if present (will be added separately)
        tags = wandb_config.pop('tags', None) or cfg.core.get('tags', [])
        if isinstance(tags, str):
            tags = [tags]
        
        # Add experiment metadata
        extra_config = {
            'model_type': cfg.model.model_type,
            'learning_rate': cfg.model.learning_rate,
            'batch_size': cfg.data.datamodule.batch_size.train,
            'cutoff': cfg.model.cutoff,
            'use_condition': cfg.model.use_condition,
        }
        
        wandb_logger = WandbLogger(
            **wandb_config,
            save_dir=save_dir,
            settings=wandb.Settings(start_method="fork"),
            tags=list(tags),
            config=extra_config,
        )
    else:
        hydra.utils.log.info("Not using <WandbLogger>")
    return wandb_logger


def get_datamodule(cfg):
    """Instantiate data module."""
    hydra.utils.log.info(f"Instantiating <{cfg.data.datamodule._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    return datamodule


def get_model(cfg_model, cfg_data, cfg_logging):
    """Instantiate model."""
    hydra.utils.log.info(f"Instantiating <{cfg_model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg_model,
        _recursive_=False,
    )
    return model


def find_ckpt(ckpt_dir) -> str | None:
    """Find latest checkpoint."""
    ckpts = list(Path(ckpt_dir).glob('*.ckpt'))
    if len(ckpts) > 0:
        ckpt_epochs = np.array([
            int(ckpt.stem.split('-')[0].split('=')[1]) 
            for ckpt in ckpts 
            if 'epoch=' in ckpt.stem
        ])
        if len(ckpt_epochs) > 0:
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
            hydra.utils.log.info(f"Found checkpoint: {ckpt}")
            return ckpt
    return None


def save_cfg(cfg, save_dir):
    """Save configuration to file."""
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (save_dir / "hparams.yaml").write_text(yaml_conf)


@hydra.main(version_base="1.3", config_path="../conf", config_name="amorphous_flow")
def train(cfg: DictConfig) -> None:
    """
    Train Amorphous Flow Matching model.
    
    :param cfg: Hydra configuration
    """
    # Hydra run directory
    run_dir = Path(HydraConfig.get().run.dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    hydra.utils.log.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Set precision and seed
    torch.set_float32_matmul_precision(cfg.train.float32_matmul_precision)
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)
    
    # Debug mode adjustments
    if cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode enabled. Forcing debugger friendly configuration!"
        )
        cfg.data.datamodule.num_workers.train = 0
        cfg.data.datamodule.num_workers.val = 0
        cfg.data.datamodule.num_workers.test = 0
        cfg.logging.wandb.mode = "offline"
    
    # Save configuration
    save_cfg(cfg, run_dir)
    
    # Instantiate data module
    datamodule = get_datamodule(cfg)
    
    # Instantiate model
    model = get_model(cfg.model, cfg.data, cfg.logging)
    
    # Print model info
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hydra.utils.log.info(f"Model: {cfg.model.model_type}")
    hydra.utils.log.info(f"Trainable parameters: {n_params:,}")
    
    # Initialize logger
    wandb_logger = get_wandb_logger(cfg, run_dir)
    loggers = [wandb_logger] if wandb_logger is not None else []
    
    # Watch model with wandb
    if wandb_logger is not None and "wandb_watch" in cfg.logging:
        hydra.utils.log.info(f"W&B watching model: {cfg.logging.wandb_watch.log}")
        wandb_logger.watch(
            model,
            log=cfg.logging.wandb_watch.log,
            log_freq=cfg.logging.wandb_watch.log_freq,
        )
    
    # Build callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg)
    
    # Instantiate trainer
    hydra.utils.log.info("Instantiating the Trainer")
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        logger=loggers,
        callbacks=callbacks,
        deterministic=cfg.train.deterministic,
        check_val_every_n_epoch=cfg.logging.val_check_interval,
        profiler=Profiler(dirpath=run_dir, filename="time_report"),
        **cfg.train.pl_trainer,
    )
    
    # Log hyperparameters
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)
    
    # Start training
    hydra.utils.log.info("Starting training!")
    ckpt = find_ckpt(run_dir)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt)
    
    # Test
    if not cfg.train.pl_trainer.fast_dev_run:
        hydra.utils.log.info("Starting testing!")
        trainer.test(datamodule=datamodule)
    
    # Close logger
    if wandb_logger is not None:
        wandb_logger.experiment.finish()
    
    hydra.utils.log.info("Training complete!")
    return trainer.callback_metrics.get('val_loss', None)


if __name__ == "__main__":
    train()

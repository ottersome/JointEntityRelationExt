import os
from logging import INFO

import lightning as L
import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from transformers import AutoTokenizer, PretrainedConfig

from entrel.data.datamodule import DataModule
from entrel.models.CopyBoi import CopyAttentionBoi
from entrel.utils import argfun, setup_logger

if __name__ == "__main__":
    args = argfun()
    logger = setup_logger("main", INFO)
    L.seed_everything(args.seed)
    meep = PretrainedConfig.from_pretrained(args.model)

    # Set Tokenizer
    logger.info("Setting up Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Data Loader
    logger.info("Instantiating DataModule")
    data_module = DataModule(args.dataset, batch_size=4)
    data_module.prepare_data(tokenizer)

    # Create Model
    # Check if path exists
    num_rels = len(data_module.metadata["relationships"])
    checkpoint_path = os.path.join(args.checkpoints, "model.ckpt")
    if os.path.exists(checkpoint_path) and not args.ignore_chkpnt:
        logger.info("📂 Loading Model from Checkpoint")
        model = CopyAttentionBoi.load_from_checkpoint(
            num_rels,
            tokenizer
            checkpoint_path,
            parent_model_name=args.model_name,
            lr=1e-5,
            dtype=args.precision,
        )
    else:
        logger.info("Loading Model from Scratch")
        model = CopyAttentionBoi(args.model, tokenizer)
    # Check memory footprint
    logger.info(f"Model's memory footprint {model.get_memory_footprint()}")

    # Do WandB Iinitalization
    if args.wandb:
        logger.info("🪄 Instantiating WandB")
        wandb_logger = WandbLogger(project=args.wandb_project_name)
    else:
        wandb_logger = None

    # Setup Trainer
    logger.info("Setting up Trainer")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        accumulate_grad_batches=4,
        max_epochs=args.epochs,
        log_every_n_steps=1,
        enable_checkpointing=True,
    )
    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="binsearch", datamodule=data_module)

    logger.info("Starting to fit model")
    trainer.fit(model, datamodule=data_module)  # type: ignore

    logger.info("Saving checkpoint")
    trainer.save_checkpoint(args.checkpoint_path)
    logger.info("✅ Done")

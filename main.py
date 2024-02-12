import os
from logging import INFO

import debugpy
import lightning as L
import torch
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from torch.autograd import set_detect_anomaly
from transformers import AutoTokenizer, BartTokenizer, PretrainedConfig

from entrel.data.datamodule import DataModule
from entrel.models.CopyBoi import CopyAttentionBoi
from entrel.utils import argfun, setup_logger

# anomaly dtection
# torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    args = argfun()
    logger = setup_logger("main", INFO)
    L.seed_everything(args.seed)
    meep = PretrainedConfig.from_pretrained(args.model)

    # If Using Debugpy then wait for connection
    if args.debug:
        logger.info("🐛 Waiting for Debugger")
        debugpy.listen(("0.0.0.0", args.port))
        debugpy.wait_for_client()
        logger.info("🐛 Debugger Connected")

    # Set Tokenizer
    logger.info("Setting up Tokenizer")
    # tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    tokenizer = BartTokenizer.from_pretrained(args.model, add_prefix_space=True)

    # Data Loader
    logger.info("Instantiating DataModule")
    data_module = DataModule(args.dataset, batch_size=4, tokenizer=tokenizer)

    # Create Model
    # Check if path exists
    num_rels = len(data_module.metadata["relationships"])
    checkpoint_path = os.path.join(args.checkpoints, "model.ckpt")
    if os.path.exists(checkpoint_path) and not args.ignore_chkpnt:
        logger.info("📂 Loading Model from Checkpoint")
        model = CopyAttentionBoi.load_from_checkpoint(
            checkpoint_path,
            data_module.metadata["relationships"],
            tokenizer,
            parent_model_name=args.model,
            lr=1e-5,
            dtype=args.precision,
        )
    else:
        logger.info("Loading Model from Scratch")
        model = CopyAttentionBoi(
            data_module.metadata["relationships"],
            tokenizer,
            parent_model_name=args.model,
            lr=1e-5,
            dtype=args.precision,
            useRemoteWeights=False,
            beam_width=10,
        )

    # Check memory footprint
    # logger.info(f"Model's memory footprint {model.get_memory_footprint()}")

    # Do WandB Initialization
    if args.wandb:
        logger.info("🪄 Instantiating WandB")
        wandb_logger = WandbLogger(
            project=args.wandb_project_name, name=args.wrname, notes=args.wrnotes
        )
    else:
        logger.info("⚠️ Starting training without wandb option.")
        wandb_logger = None

    # CheckPointing
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints", save_top_k=3, monitor="val_loss"
    )
    # Setup Trainer
    logger.info("Setting up Trainer")
    trainer = L.Trainer(
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        accumulate_grad_batches=4,
        max_epochs=args.epochs,
        val_check_interval=0.05,
        log_every_n_steps=1,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
    )

    tuner = Tuner(trainer)
    tuner.scale_batch_size(model, mode="binsearch", datamodule=data_module)

    logger.info("Starting to fit model")
    trainer.fit(model, datamodule=data_module)  # type: ignore

    logger.info("Saving checkpoint")
    trainer.save_checkpoint(args.checkpoint_path)
    logger.info("✅ Done")

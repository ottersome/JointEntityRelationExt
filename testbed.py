"""
Script is meant to load a lightning checkpoint and 
it with a test dataset.
Application is a Bart model outputting in an autoregressive manner.
"""
import argparse
import json
import os
import random
from logging import INFO

import debugpy
import lightning as L
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner.tuning import Tuner
from transformers import AutoTokenizer, BartConfig, BartTokenizer, PretrainedConfig

from entrel.data.datamodule import parse_webnlg_ds
from entrel.models.CopyBoi import CopyAttentionBoi
from entrel.utils import TokenType, argfun, setup_logger

logger = setup_logger("main_testbed", INFO)
# Set all seeds
L.seed_everything(42)


def afun():
    af = argparse.ArgumentParser()
    af.add_argument("--chkpnt_path", required=True)
    af.add_argument("--metadata_path", default="./.cache/metadata.json")
    af.add_argument("--samples", default=1, type=int)
    af.add_argument("-d", "--debug", action="store_true", default=False)
    af.add_argument("-p", "--port", default=42019, type=int)
    return af.parse_args()


args = afun()
if args.debug:
    logger.info("🐛 Waiting for Debugger")
    debugpy.listen(("0.0.0.0", args.port))
    debugpy.wait_for_client()
    logger.info("🐛 Debugger Connected")

#  Load Tokenizer


tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

bart_config = BartConfig.from_pretrained("facebook/bart-large")
max_inpt_len = bart_config.max_position_embeddings

max_output = 256  # This is a value I manually established after looking at data

# Load the json file
metadata = {}
with open(args.metadata_path, "r") as f:
    metadata = json.load(f)

seen_rels = {k: i for i, k in enumerate(metadata["relationships"])}
num_relationships = len(metadata["relationships"])
logger.info(f"Working with {num_relationships} relationships")


# Vocab Size
vocab_size = tokenizer.vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
model = CopyAttentionBoi.load_from_checkpoint(
    args.chkpnt_path,
    relationship_list=metadata["relationships"],
    tokenizer=tokenizer,
    parent_model_name="facebook/bart-large",
    lr=1e-5,
    dtype=torch.float16,
    useRemoteWeights=False,
    beam_width=3,
).to(device)
model.eval()

all_tests = []
# Load Dataset
if os.path.exists("./.cache/test.parquet"):
    df = pd.read_parquet("./.cache/test.parquet")
    logger.info("Found cached test.parquet. Using it")
    # Conver to simple list
    for i in df.index:
        all_tests.append(
            [
                np.array(df.loc[i, "tokens"]).tolist(),
                np.array(df.loc[i, "triplets"]).tolist(),
                np.array(df.loc[i, "token_types"]).tolist(),
            ]
        )
else:
    logger.info("Did not find test.parquet, downloading...")
    dataset = load_dataset("web_nlg", "release_v3.0_en")
    ds_test = dataset["test"]  # type : ignore
    all_tests, _ = parse_webnlg_ds(
        ds_test, "test", tokenizer, seen_rels, add_RathThan_Rem=False
    )

# Pivotal Info
vocab_size = tokenizer.vocab_size
relationships = metadata["relationships"]
num_relationships = len(metadata["relationships"])

# Sample randomly from all_tests
samples = random.sample(all_tests, args.samples)

tset_tokens = torch.empty((args.samples, max_inpt_len), dtype=torch.long, device=device)
target = torch.empty((args.samples, max_output), dtype=torch.long, device=device)
token_types = torch.LongTensor()
for i, sample in enumerate(samples):
    tset_tokens[i, :] = torch.LongTensor(sample[0])  # Original String
    target[i, :] = torch.LongTensor(sample[1])

    token_types = torch.LongTensor(sample[2])
    target[i, token_types == TokenType.RELATIONSHIP] += vocab_size
    target[i, token_types == TokenType.COPY] += vocab_size + num_relationships

# Create Attention Masks based on pad_token_id (not needed tbh)
attn_mask = torch.where(tset_tokens != tokenizer.pad_token_id, 1, 0)
# Feed The Model and Get Responses
estimations = model(tset_tokens, attn_mask)

logger.info("Evaluation Report: " f"- We have evaluated {len(samples)} samples")
for i in range(estimations.size(1)):
    # First Padding Indices
    fpi_est = torch.where((estimations[0, i, :] == tokenizer.pad_token_id)[0])
    fpi_targ = torch.where((target[0:] == tokenizer.pad_token_id)[0])
    if len(fpi_est) == 0:
        fpi_est = -1  # This is not great but fix for now

    est_mask = estimations[0].new_zeros(size=estimations.shape[1:])
    est_mask[est_mask >= tokenizer.vocab_size] = 1
    est_mask[est_mask >= tokenizer.vocab_size + num_relationships] = 2

    # Decode
    est_dec = model.multi_decode(
        tset_tokens[0], estimations[0, i, : fpi_est[0]], est_mask, relationships
    )
    est_targ = model.multi_decode(
        tset_tokens[0], target[0, : fpi_targ[0]], token_types, relationships
    )
    id = "{:2}".format(i)
    logger.info(f"{id}) Estimation : {est_dec}" f"      Target : {est_targ}\n")

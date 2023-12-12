import logging
import os
from argparse import ArgumentParser
from enum import Enum

from torch import float16


# Create an enum of types of datasets
class DatasetInUse(Enum):
    NLG = 1
    NYC = 1


def argfun():
    ap = ArgumentParser()
    ap.add_argument("--dataset", default="NLG")
    ap.add_argument("--checkpoints", default="./checkpoints")
    ap.add_argument("--seed", default=420)
    ap.add_argument("--ignore_chkpnt", action="store_false", default=True)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb_project_name", default="JointEntRel_Extraction")
    ap.add_argument("--epochs", default=1)
    ap.add_argument("--precision", default=float16)
    ap.add_argument("--model", default="facebook/bart-large")

    args = ap.parse_args()
    ## Sanitize
    # Dataset
    if args.dataset == "NLG":
        args.dataset = DatasetInUse.NLG
    elif args.dataset == "NYC":
        args.dataset = DatasetInUse.NYC
    else:
        raise ValueError("Invalid dataset")
    # Checkpoints
    os.makedirs(args.checkpoints, exist_ok=True)

    return args


def setup_logger(name="main", level=logging.INFO):
    # Setup Parent Directory
    os.makedirs("logs/", exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    # Create Handlers
    fh = logging.FileHandler(os.path.join("logs/", name + ".log"))
    sh = logging.StreamHandler()
    fh.setLevel(logging.DEBUG)
    sh.setLevel(level)
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger

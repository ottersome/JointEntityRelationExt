"""
This is based off of Zheng's https://github.com/xiangrongzeng/multi_re
But just using a pretrained attention BART 
"""

from logging import INFO
from pathlib import Path

import lightning as L
import torch
from torch.nn import Linear, NLLLoss
from transformers import AutoModel, BartConfig, BartModel, PretrainedConfig

from ..utils import setup_logger


class CopyAttentionBoi(L.LightningModule):
    def __init__(
        self,
        amount_of_relations: int,
        parent_model_name="facebook/bart-large",
        lr=1e-5,
        dtype=torch.float16,
        useRemoteWeights=True,
    ):
        """
        Arguments
        ---------
            parent_model_name: Name of model to use from HuggingFace's repertoire,
            lr: learning rate
            dtype: precision for model, we'll likely need float16 for smaller gpus
            useParentWeights: whether or not to load a checkpoint or just used weigths provided by 🤗

        """
        super().__init__()

        self.my_logger = setup_logger("CopyAttentionBoi", INFO)

        self.loss = NLLLoss()  # TODO: make sure this is alright
        # Create Configuration as per Parent
        self.bart_config = BartConfig.from_pretrained(parent_model_name)

        if useRemoteWeights:
            self.base = BartModel.from_pretrained(parent_model_name)
        else:  # Only Load the Structure
            self.base = BartModel(self.bart_config)

        self.rel_head = Linear(self.bart_config.d_model, amount_of_relations)
        self.indexing_head = Linear(
            self.bart_config.d_model, self.config.max_position_embeddings
        )
        # config = BartConfig.from_pretrained(parent_model_name)
        # if useRemoteWeights:
        # self.model = TripleBartWithCopyMechanism.from_pretrained(
        # parent_model_name,
        # amount_of_relations,
        # config=config,
        # )
        # else:
        # self.model = TripleBartWithCopyMechanism(config, amount_of_relations)
        # elif checkpoidnt_path_obj.exists():
        # self.model = TripleBartWithCopyMechanism(config, amount_of_relations, True)

    def forward(self, x, attention_mask):
        """
        For inference
        """
        return self.model(x, attention_mask=attention_mask)  # CHECK: is this correct?

    def training_step(self, batches):
        self.my_logger.info("")
        yhat = self.model(batches)
        loss = self.loss(yhat)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Validation")
        return bar


class TripleBartWithCopyMechanism(BartModel):
    def __init__(self, config: PretrainedConfig, amount_of_relations):
        super().__init__(config)
        self.head = Linear(config.d_model, amount_of_relations)
        # if not use_parent_weights:#Load from checkpoint #TODO: Load from checkpoint

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids)
        y = self.head(h)
        return y

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path, amount_of_relations, *model_args, **kwargs
    ):
        kwargs["amount_of_relatations"] = amount_of_relations
        model = super(TripleBartWithCopyMechanism, cls).from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        output_size = (
            amount_of_relations + model.config.max_position_embeddings + 1
        )  # 1 for ??
        # Set new head
        model.head = Linear(model.config.d_model, output_size)

        return model

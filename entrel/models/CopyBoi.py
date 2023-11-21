"""
This is based off of Zheng's https://github.com/xiangrongzeng/multi_re
But just using a pretrained attention BART 
"""

from logging import INFO

import lightning as L
import torch
from torch.nn import Linear, NLLLoss
from transformers import AutoModel, BartConfig, BartModel, PretrainedConfig

from ..utils import setup_logger


class CopyAttentionBoi(L.LightningModule):
    def __init__(
        self,
        amount_of_relations,
        parent_model_name="facebook/bart-large",
        lr=1e-5,
        dtype=torch.float16,
        useParentWeights=True,
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
        config = BartConfig.from_pretrained(parent_model_name)

        if useParentWeights:
            self.model = TripleBartWithCopyMechanism.from_pretrained(
                parent_model_name,
                config=config,
                amount_of_relations=amount_of_relations,
                use_parent_weights=True,
            )
        else:
            self.model = TripleBartWithCopyMechanism(config, amount_of_relations, True)

    def forward(self, x, attention_mask):
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
    def __init__(
        self,
        config: PretrainedConfig,
        amount_of_relations,
        use_parent_weights=True,
    ):
        super().__init__(config)
        output_size = amount_of_relations + self.config.max_position_embeddings + 1
        self.head = Linear(self.config.d_model, output_size)
        # if not use_parent_weights:#Load from checkpoint #TODO: Load from checkpoint

    def forward(self, input_ids, attention_mask):
        h = self.model(input_ids)
        y = self.head(h)
        return y

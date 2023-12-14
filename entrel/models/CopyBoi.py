"""
This is based off of Zheng's https://github.com/xiangrongzeng/multi_re
But just using a pretrained attention BART 
"""

from logging import INFO
from pathlib import Path

import lightning as L
import torch
import torch.nn.functional as F
from torch.nn import Linear, NLLLoss
from transformers import (
    AutoModel,
    BartConfig,
    BartModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from ..utils import setup_logger


class CopyAttentionBoi(L.LightningModule):
    def __init__(
        self,
        amount_of_relations: int,
        tokenizer: PreTrainedTokenizer,
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

        self.tokenizer = tokenizer

        self.my_logger = setup_logger("CopyAttentionBoi", INFO)

        self.loss = NLLLoss()  # TODO: make sure this is alright
        # Create Configuration as per Parent
        self.bart_config = BartConfig.from_pretrained(parent_model_name)

        if useRemoteWeights:
            self.base = BartModel.from_pretrained(parent_model_name)
        else:  # Only Load the Structure
            self.base = BartModel(self.bart_config)

        assert isinstance(self.base, BartModel)
        # Using `load_checkpoint` outside of the model would load the weights that are
        # not yet loaded thus far.
        self.copy_head = Linear(
            self.bart_config.d_model, self.bart_config.d_model
        )  # Encoder States Size x Decoder State Sizej
        self.vocabulary_head = Linear(
            self.bart_config.d_model, self.bart_config.vocab_size
        )

    def forward(self, batchx, attention_mask):
        """
        For inference, also for guessing batch size
        """
        # Now We do the painful part of doing sequential inference
        # TODO: do it overall
        padding_token = self.tokenizer.pad_token_id
        attn_mask = torch.ones_like(batchx)
        attn_mask[batchx == padding_token] = 0
        encoder_states = self.base.encoder(batchx, attn_mask)

        # Use decoder for inference
        outputs = torch.full(
            (batchx.shape[0], 1), self.tokenizer.convert_tokens_to_ids("<s>")
        )
        for i in range(batchx.shape[1]):
            attn_mask = torch.ones_like(outputs)
            attn_mask[outputs == padding_token] = 0
            decoder_states = self.base.decoder(outputs, encoder_states, attn_mask)

            # Pass Decoder States
            probs = self._mixed_softmax(decoder_states)

        return

    def _mixed_softmax(self, encoder_states, decoder_states):
        copy_scores = decoder_states @ F.relu(self.copy_head(encoder_states))

        vocab_scores = self.vocabulary_head(decoder_states)

    def training_step(self, batches, batch_idx):
        assert isinstance(self.model, BartModel)
        self.my_logger.debug(f"Going through batch idx {batch_idx}")
        inputs, target = batches
        padding_token = self.tokenizer.pad_token_id
        sep_token = self.tokenizer.sep_token

        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == padding_token] = 0
        encoder_outputs, _, _ = self.base.encoder(inputs, encoder_attn_mask)
        # TODO:  Create attention mask where padding tokens. Otherwise padding tokens dont work

        decoder_attn_mask = torch.ones_like(target)
        # TODO : We need to figureout how it is that the targets are batched up.( I guess we could batch them in preprocesing and also by the datamodule).
        # We'll see when it complains
        decoder_attn_mask[target == padding_token] = 0
        decoder_hidden_outputs = self.base.decoder(target, encoder_outputs)

        first_indices_of_pad = find_padding_indices(target, padding_token)
        # Get Indices of Relationships

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


def find_padding_indices(tensor, padding_token=0):
    indices = []
    for i in range(tensor.size(0)):
        try:
            index = (tensor[i] == padding_token).nonzero(as_tuple=True)[0][0]
        except IndexError:  # no padding token in this row
            index = -1
        indices.append(index)
    return indices

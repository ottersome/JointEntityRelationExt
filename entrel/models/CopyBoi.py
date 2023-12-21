"""
This is based off of Zheng's https://github.com/xiangrongzeng/multi_re
But just using a pretrained attention BART 
"""

from logging import INFO
from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, NLLLoss
from transformers import (AutoModel, BartConfig, BartModel, PretrainedConfig,
                          PreTrainedTokenizer)

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
        beam_width=10,
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
        self.beam_width = beam_width

        self.my_logger = setup_logger("CopyAttentionBoi", INFO)

        self.loss = NLLLoss()  # TODO: make sure this is alright
        # Create Configuration as per Parent
        self.bart_config = BartConfig.from_pretrained(parent_model_name)

        if useRemoteWeights:
            self.base = BartModel.from_pretrained(parent_model_name)
        else:  # Only Load the Structure
            self.base = BartModel(self.bart_config) # type: ignore

        assert isinstance(self.base, BartModel)
        # Using `load_checkpoint` outside of the model would load the weights that are
        # not yet loaded thus far.
        self.copy_head = Linear(
            self.bart_config.d_model, self.bart_config.d_model
        )  # Encoder States Size x Decoder State Sizej
        self.relationship_head = Linear(self.bart_config.d_model, amount_of_relations)

    def forward(self, batchx, attention_mask):
        """
        For inference, also for guessing batch size
        """
        # Check if it is training
        padding_token = self.tokenizer.pad_token_id
        attn_mask = torch.ones_like(batchx)
        attn_mask[batchx == padding_token] = 0
        encoder_states = self.base.encoder(batchx, attn_mask) # type: ignore

        # Use decoder for inference
        if not self.traning:
            self._autoregressive_decoder(encoder_states)
        else:
            pass #TODO: write teacher-forcing method here

        return

    def _autoregressive_decoder(self, encoder_states):
        # Start the memory
        outputs = torch.full(
            (encoder_states.shape[0], 1), int(self.tokenizer.convert_tokens_to_ids("<s>"))
        )
        self._beamsearch(encoder_states, outputs)

    def _mixed_softmax(self, encoder_states, decoder_states) -> torch.Tensor:
        # CHECK: Will likely have to do bmm here to consider batches
        copy_scores = decoder_states @ F.relu(self.copy_head(encoder_states))

        rel_scores = self.vocabulary_head(decoder_states)
        concat = torch.cat((copy_scores, rel_scores), dim=1)
        return concat

    def _beamsearch(
        self,
        encoder_states: torch.Tensor,
        initial_states: torch.Tensor,
    ):
        # We keep topk paths as such:
        batch_size = initial_states.shape[0]
        pad_token = self.tokenizer.pad_token_id
        encoder_attn_mask = torch.ones_like(encoder_states)
        encoder_attn_mask[encoder_states == pad_token] = 0

        # Initiation
        # (batch_size x beam width) x (sequence lengt)
        # Initial states is just (batch_size) x (sequence length)
        # we want to repeat sequences in a new dimension 1 for (beam_width)
        cur_state = torch.repeat_interleave(initial_states, self.beam_width, dim=0)

        cur_seq_length = 1
        cur_sequences = initial_states
        for s in self.decoder.max_seq_length:
            
            cs_tensor = torch.tensor(cur_sequences)
            cs_attn_mask = cs_tensor.new_ones()
            cs_attn_mask[cs_tensor == pad_token] = 0
            decoder_states = self.decoder(
                input_ids=cs_tensor,
                attention_mask=cs_attn_mask,
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=encoder_attn_mask,
            )
            probabilities = self._mixed_softmax(encoder_states, decoder_states)

            top_p, top_i = torch.topk(probabilities, self.beam_width)
            top_i = top_i.view(batch_size, self.beam_width, self.beam_width, 1).unsqueeze(-1)
            # cur_sate is (batch_size x beam_width ) x (sequence length) in shape
            # probabilities are (batch_size x beam_width) x (beam_width)
            # We want a cartesian product  only of matching indices on (batch_size x beam_width)
            new_view = cur_state.view(batch_size, self.beam_width, cur_seq_length)
            expanded_view = new_view.unsqueeze(2).expand(-1, -1, self.beam_width, -1)
            candidates = torch.cat((expanded_view, top_i), dim=-1)

            completed_bool_tape = [ for ]

            # Make it back into (batch_size) x ()
            cur_seq_length += 1
        return

    def training_step(self, batches, batch_idx):
        assert isinstance(self.base, BartModel)
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

        yhat = self.base(batches)
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

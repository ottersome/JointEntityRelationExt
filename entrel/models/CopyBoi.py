"""
This is based off of Zheng's https://github.com/xiangrongzeng/multi_re
But just using a pretrained attention BART 
"""

from logging import INFO
from pathlib import Path
from typing import List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, NLLLoss
from transformers import (
    AutoModel,
    BartConfig,
    BartModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)

from ..utils import TokenType, setup_logger


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
        self.amount_of_relations = amount_of_relations

        self.my_logger = setup_logger("CopyAttentionBoi", INFO)

        # CHECK: amount_of_relations has correct value
        # self.criterion = TypedNLLLoss(tokenizer.vocab_size, self.amount_of_relations)
        self.criterion = nn.NLLLoss()
        # Create Configuration as per Parent
        self.bart_config = BartConfig.from_pretrained(parent_model_name)

        if useRemoteWeights:
            self.base = BartModel.from_pretrained(parent_model_name)
        else:  # Only Load the Structure
            self.base = BartModel(self.bart_config)  # type: ignore

        assert isinstance(self.base, BartModel)

        ####################
        # Heads
        ####################
        self.my_logger.info(
            f"Setting up the relationship head with {self.amount_of_relations}"
        )
        self.copy_head = Linear(
            self.bart_config.d_model, self.bart_config.d_model
        )  # Encoder States Size x Decoder State Sizej
        self.relationship_head = Linear(
            self.bart_config.d_model, self.amount_of_relations
        )
        # Just the head for vocabulary
        self.normal_head = Linear(self.bart_config.d_model, self.bart_config.vocab_size)

    def forward(self, batchx, attention_mask):
        """
        For inference, also for guessing batch size
        """
        # Check if it is training
        padding_token = self.tokenizer.pad_token_id
        attn_mask = torch.ones_like(batchx)
        attn_mask[batchx == padding_token] = 0
        encoder_states = self.base.encoder(batchx, attn_mask)  # type: ignore

        # Use decoder for inference
        if not self.traning:
            self._autoregressive_decoder(encoder_states)
        else:
            pass  # TODO: write teacher-forcing method here

        return

    def _autoregressive_decoder(self, encoder_states):
        # Start the memory
        outputs = torch.full(
            (encoder_states.shape[0], 1),
            int(self.tokenizer.convert_tokens_to_ids("<s>")),
        )
        self._beamsearch(encoder_states, outputs)

    def _mixed_logsoftmax(self, encoder_states, decoder_states) -> torch.Tensor:
        """
        returns
        -------
            probabilities: (batch_length) x (target_len) x ( vocab_length + relationship_lengths + encoder_states_len)
        """
        # encoder_states is (batch_length) x (sequence_length) x (hidden_dim)
        encoder_hdn_transformed = self.copy_head(encoder_states)
        copy_scores = torch.bmm(decoder_states, encoder_hdn_transformed.transpose(1, 2))
        # copy_scores = decoder_states @ self.copy_head(encoder_states)
        # CHECK: if we have to use sqrt normalization in a simliar way to attention for balancing gradients

        vocab_scores = self.normal_head(decoder_states)
        rel_scores = self.relationship_head(decoder_states)

        all_scores = torch.cat((vocab_scores, rel_scores, copy_scores), dim=-1)

        probabilities = F.log_softmax(all_scores, dim=-1)

        return probabilities

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

            decoder_batch = self.decoder(
                input_ids=cs_tensor,
                attention_mask=cs_attn_mask,
                encoder_hidden_states=encoder_states,
                encoder_attention_mask=encoder_attn_mask,
            )
            probabilities = self._mixed_logsoftmax(encoder_states, decoder_batch)

            top_p, top_i = torch.topk(probabilities, self.beam_width)
            top_i = top_i.view(
                batch_size, self.beam_width, self.beam_width, 1
            ).unsqueeze(-1)
            # cur_sate is (batch_size x beam_width ) x (sequence length) in shape
            # probabilities are (batch_size x beam_width) x (beam_width)
            # We want a cartesian product  only of matching indices on (batch_size x beam_width)
            new_view = cur_state.view(batch_size, self.beam_width, cur_seq_length)
            expanded_view = new_view.unsqueeze(2).expand(-1, -1, self.beam_width, -1)
            candidates = torch.cat((expanded_view, top_i), dim=-1)

            # completed_bool_tape = [ for ]

            # Make it back into (batch_size) x ()
            cur_seq_length += 1
        return

    def training_step(self, batches, batch_idx):
        assert isinstance(self.base, BartModel)
        # self.my_logger.debug(f"Going through batch idx {batch_idx}")
        inputs, target, ref_text, ref_triplets = batches
        padding_token = self.tokenizer.pad_token_id

        # Generates Masks
        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == padding_token] = 0
        encoder_outputs = self.base.encoder(inputs, encoder_attn_mask).last_hidden_state

        decoder_attn_mask = torch.ones_like(target)
        decoder_attn_mask[target == padding_token] = 0
        decoder_hidden_outputs = self.base.decoder(
            target, decoder_attn_mask, encoder_outputs, encoder_attn_mask
        ).last_hidden_state

        mixed_probabilities = self._mixed_logsoftmax(
            encoder_outputs, decoder_hidden_outputs
        )

        flat_probabilities = mixed_probabilities.view(-1, mixed_probabilities.size(-1))
        flat_target = target.view(-1)

        # loss = self.criterion(mixed_probabilities, target, masks)
        loss = self.criterion(flat_probabilities, flat_target)
        loss_avg = loss.mean()
        self.log(
            "train_loss", loss_avg.item(), prog_bar=True, on_step=True, on_epoch=True
        )
        # self.my_logger.debug(
        #     f"At batch {batch_idx} we are looking at ref_text {ref_text}  and triplets {ref_triplets}"
        # )
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description("Validation")
        return bar

    def validation_step(self, ref_batches: List[Tensor], batch_idx):
        # Whole of validation is here:
        inputs, target, ref_text, ref_triplets = ref_batches
        padding_token = self.tokenizer.pad_token_id
        sep_token = self.tokenizer.sep_token

        # Generates Masks

        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == padding_token] = 0
        encoder_outputs = self.base.encoder(inputs, encoder_attn_mask).last_hidden_state

        decoder_attn_mask = torch.ones_like(target)
        decoder_attn_mask[target == padding_token] = 0
        decoder_hidden_outputs = self.base.decoder(
            target, decoder_attn_mask, encoder_outputs, encoder_attn_mask
        ).last_hidden_state

        mixed_probabilities = self._mixed_logsoftmax(
            encoder_outputs, decoder_hidden_outputs
        )

        flat_probabilities = mixed_probabilities.view(-1, mixed_probabilities.size(-1))
        flat_target = target.view(-1)

        # loss = self.criterion(mixed_probabilities, target, masks)
        loss = self.criterion(flat_probabilities, flat_target)
        loss_avg = loss.mean()
        self.log(
            "val_loss", loss_avg.item(), prog_bar=True, on_step=True, on_epoch=True
        )
        self.my_logger.debug(
            f" At validation batch_idx {batch_idx} we have examples ref_text {ref_text} and triplets {ref_triplets}"
        )
        return loss


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


class TypedNLLLoss(nn.Module):
    def __init__(self, vocab_size: int, rel_space_size: int):
        super(TypedNLLLoss, self).__init__()
        self.vocab_size = vocab_size
        self.rel_space_size = rel_space_size
        self.nll_loss = nn.NLLLoss(reduction="none")

    def forward(self, output, target, masks):
        # Split the output and target into normal and copy parts
        third_dim_len = output.size(2)
        output_normal = output[
            masks["normal"].unsqueeze(-1).expand(-1, -1, third_dim_len),
        ][: self.vo]
        output_rel = output[
            :, masks["relationship"], self.vocab_size : self.rel_space_size
        ]
        output_copy = output[:, masks["copy"], self.vocab_size + self.rel_space_size :]

        target_normal = target[masks["normal"]]
        target_rel = target[masks["relationship"]]
        target_copy = target[masks["copy"]]

        # Compute the loss for the normal and copy parts separately
        loss_normal = self.nll_loss(output_normal, target_normal)
        loss_rel = self.nll_loss(output_rel, target_rel)
        loss_copy = self.nll_loss(output_copy, target_copy)
        # Combine the losses
        loss = loss_normal + loss_copy + loss_rel
        # Average the loss over the non-zero elements
        return loss.sum()

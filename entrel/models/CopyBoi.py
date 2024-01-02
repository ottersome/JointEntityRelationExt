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
        # Increase Embedding for new tokens
        max_input_len = self.bart_config.max_position_embeddings
        self.base.resize_token_embeddings(
            # self.tokenizer.vocab_size + self.amount_of_relations + max_input_len
            self.tokenizer.vocab_size
            + self.amount_of_relations
        )

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

    def forward(self, batch, attention_mask):
        """
        For inference, also for guessing batch size
        """
        # Check if it is training
        inputs = batch

        # Use decoder for inference
        if not self.training:
            return self._autoregressive_decoder(inputs)
        else:
            pass  # TODO: write teacher-forcing method here

        return

    def _autoregressive_decoder(self, inputs: torch.Tensor):
        # Start the memory
        outputs = torch.full(  # Initial State
            (inputs.shape[0], 1),
            int(self.tokenizer.convert_tokens_to_ids("<s>")),
        ).to(inputs.device)

        return self._beamsearch(inputs, outputs)

    def _mixed_logits(self, encoder_states, decoder_states) -> torch.Tensor:
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

        # probabilities = F.log_softmax(all_scores, dim=-1)

        return all_scores

    def _beamsearch(
        self,
        inputs: torch.Tensor,
        initial_states: torch.Tensor,
    ):
        # We keep topk paths as such:
        batch_size = initial_states.shape[0]
        pad_token = self.tokenizer.pad_token_id

        # Input Stuff
        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == pad_token] = 0
        encoder_output = self.base.encoder(inputs, encoder_attn_mask)  # type: ignore
        encoder_states = encoder_output.last_hidden_state

        # Initiation
        # (batch_size x beam width) x (sequence lengt)
        # Initial states is just (batch_size) x (sequence length)
        # we want to repeat sequences in a new dimension 1 for (beam_width)
        # cur_state = torch.repeat_interleave(initial_states, self.beam_width, dim=0)

        cur_seq_length = 1
        # (batch_size) x (beam_width) x (cur_seq_length)
        cur_sequences = initial_states.unsqueeze(-1)
        # OPTIM: we could store the decoder states
        probabilities = torch.zeros(
            (cur_sequences.size(0), cur_sequences.size(1)), device=initial_states.device
        )

        for s in range(256):  # type: ignore
            cur_beam_width = cur_sequences.size(1)

            ########################################
            # Organize and get probabilities of current choices
            ########################################

            cs_tensor = cur_sequences.view(batch_size * cur_beam_width, -1)
            rptd_enc_states = encoder_states.repeat_interleave(cur_beam_width, dim=0)
            rptd_enc_attn = encoder_attn_mask.repeat_interleave(cur_beam_width, dim=0)

            cs_attn_mask = cs_tensor.new_ones(cs_tensor.shape)
            cs_attn_mask[cs_tensor == pad_token] = 0

            decoder_output = self.base.decoder(  # type: ignore
                input_ids=cs_tensor,
                attention_mask=cs_attn_mask,
                encoder_hidden_states=rptd_enc_states,
                encoder_attention_mask=rptd_enc_attn,
            )

            decoder_hiddstates = decoder_output.last_hidden_state
            dhss = decoder_hiddstates.shape

            decoder_last_hidd_state = decoder_hiddstates[:, -1, :].view(
                dhss[0], 1, dhss[2]
            )
            mixed_logits = self._mixed_logits(rptd_enc_states, decoder_last_hidd_state)
            mixed_probabilities = F.softmax(mixed_logits, dim=-1)

            cum_probs = probabilities.view(batch_size * cur_beam_width, -1, 1) + (
                -1 * mixed_probabilities.log()
            )

            # Reconstruct to do topk across beams
            proper_probs = cum_probs.view(batch_size, -1)

            prob_state_space_size = (
                self.tokenizer.vocab_size
                + self.amount_of_relations
                + self.bart_config.max_position_embeddings
            )

            top_p, top_i = torch.topk(proper_probs, self.beam_width)
            top_i = (top_i % prob_state_space_size).view(batch_size, self.beam_width, 1)

            probabilities = top_p

            ## Fix Copy:
            copy_mask = (
                top_i >= self.tokenizer.vocab_size + self.amount_of_relations
            )  # .nonzero(as_tuple=False)

            ids = top_i[copy_mask] % (
                self.tokenizer.vocab_size + self.amount_of_relations
            )
            batch_idx = copy_mask.nonzero(as_tuple=True)
            if len(batch_idx) != 0:  # If there are ids to replace
                enc_input_vals = inputs[batch_idx[0], ids]
                top_i[copy_mask] = enc_input_vals

            # cur_sate is (batch_size x beam_width ) x (sequence length) in shape
            # probabilities are (batch_size x beam_width) x (beam_width)
            # We want a cartesian product  only of matching indices on (batch_size x beam_width)
            if cur_sequences.size(1) != self.beam_width:
                cur_sequences = cur_sequences.repeat_interleave(self.beam_width, dim=1)

            cur_sequences = torch.cat((cur_sequences, top_i), dim=-1)
            # new_view = cur_state.view(batch_size, self.beam_width, cur_seq_length)
            # expanded_view = new_view.unsqueeze(2).expand(-1, -1, self.beam_width, -1)
            # candidates = torch.cat((expanded_view, top_i), dim=-1)

            # completed_bool_tape = [ for ]

            # Make it back into (batch_size) x ()
            cur_seq_length += 1
        self.my_logger.info(f"Returning cur_sequences with shape {cur_sequences.shape}")
        return cur_sequences

    def _teacher_forcing_inputs(
        self,
        encoder_input: torch.Tensor,
        hybrid_targets: torch.Tensor,
        token_types: torch.Tensor,
    ) -> torch.Tensor:
        vocab_size = self.tokenizer.vocab_size
        amnt_rels = self.amount_of_relations
        # Change Values
        mask_rel = token_types == TokenType.RELATIONSHIP.value
        mask_copy = token_types == TokenType.COPY.value
        vocab_target = hybrid_targets.clone()

        # Replace
        for i in range(hybrid_targets.size(0)):
            idxs = torch.where(mask_copy[i, :])[0]
            encoder_seq_idx = hybrid_targets[i, idxs]
            encoder_vals = encoder_input[i, encoder_seq_idx]
            vocab_target[i, idxs] = encoder_vals

        vocab_target[mask_rel] += vocab_size  # Remember to increase emebdding size

        hybrid_targets[mask_rel] += vocab_size
        hybrid_targets[mask_copy] += vocab_size + amnt_rels

        return vocab_target

    def training_step(self, batches, batch_idx):
        assert isinstance(self.base, BartModel)
        # self.my_logger.debug(f"Going through batch idx {batch_idx}")
        inputs, hybrid_target, token_types = batches
        padding_token = self.tokenizer.pad_token_id

        # Create Teacher Forcing Decoder Inputs
        vocab_target = self._teacher_forcing_inputs(inputs, hybrid_target, token_types)
        # TODO: replace `target` with ready `vocab_target`

        # Generates Masks
        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == padding_token] = 0
        encoder_outputs = self.base.encoder(inputs, encoder_attn_mask).last_hidden_state

        decoder_attn_mask = torch.ones_like(vocab_target)
        decoder_attn_mask[vocab_target == padding_token] = 0
        decoder_hidden_outputs = self.base.decoder(
            vocab_target, decoder_attn_mask, encoder_outputs, encoder_attn_mask
        ).last_hidden_state

        mixed_logits = self._mixed_logits(encoder_outputs, decoder_hidden_outputs)
        mixed_logsofts = F.log_softmax(mixed_logits, dim=-1)

        flat_probabilities = mixed_logsofts.view(-1, mixed_logsofts.size(-1))
        # Hybrid T
        flat_target = hybrid_target.view(-1)

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
        inputs, target, token_types = ref_batches
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

        mixed_logits = self._mixed_logits(encoder_outputs, decoder_hidden_outputs)
        mixed_logsofts = F.log_softmax(mixed_logits, dim=-1)

        flat_probabilities = mixed_logsofts.view(-1, mixed_logsofts.size(-1))
        flat_target = target.view(-1)

        # loss = self.criterion(mixed_probabilities, target, masks)
        loss = self.criterion(flat_probabilities, flat_target)
        loss_avg = loss.mean()
        self.log(
            "val_loss", loss_avg.item(), prog_bar=True, on_step=True, on_epoch=True
        )
        # self.my_logger.debug(
        #     f" At validation batch_idx {batch_idx} we have examples ref_text {ref_text} and triplets {ref_triplets}"
        # )
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

"""
This is based off of Zheng's https://github.com/xiangrongzeng/multi_re
But just using a pretrained attention BART 
"""

from logging import INFO
from pathlib import Path
from typing import Dict, List

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
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
        relationship_list: Dict[str, int],
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
        # Invert the dictionary to have integer keys
        self.rel_list = {v: k for k, v in relationship_list.items()}
        self.amount_of_relations = len(relationship_list)

        self.my_logger = setup_logger("CopyAttentionBoi", INFO)
        self.padding_token = tokenizer.pad_token_id

        self.criterion = nn.NLLLoss(ignore_index=self.padding_token)
        # self.criterion = nn.NLLLoss()
        # Create Configuration as per Parent
        self.bart_config = BartConfig.from_pretrained(parent_model_name)

        if useRemoteWeights:
            self.base = BartModel.from_pretrained(parent_model_name)
        else:  # Only Load the Structure
            self.base = BartModel(self.bart_config)  # type: ignore

        # Freeeze most of the base, except embedding
        for param in self.base.parameters():
            param.requires_grad = False
        for param in self.base.shared.parameters():
            param.requires_grad = True

        assert isinstance(self.base, BartModel)
        # Increase Embedding for new tokens
        max_input_len = self.bart_config.max_position_embeddings
        self.old_vocab_size = self.tokenizer.vocab_size
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
        self.copy_head_nonlinear = torch.nn.ReLU()
        self.relationship_head = Linear(
            self.bart_config.d_model, self.amount_of_relations
        )
        # Just the head for vocabulary
        self.normal_head = Linear(
            self.bart_config.d_model, self.tokenizer.vocab_size
        )  # self.bart_config.vocab_size)

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
        # encoder_hdn_transformed = self.copy_head_nonlinear(
        #     (self.copy_head(encoder_states))
        # )
        encoder_hdn_transformed = self.copy_head(encoder_states)

        # TODO:  Probably have to pass above through a non-linear before computing copy_scores.
        copy_scores = torch.bmm(decoder_states, encoder_hdn_transformed.transpose(1, 2))
        # copy_scores = decoder_states @ self.copy_head(encoder_states)
        # CHECK: if we have to use sqrt normalization in a simliar way to attention for balancing gradients

        vocab_scores = self.normal_head(decoder_states)
        rel_scores = self.relationship_head(decoder_states)

        all_scores = torch.cat((vocab_scores, rel_scores, copy_scores), dim=-1)
        # all_scores = torch.cat((vocab_scores, copy_scores), dim=-1)

        # probabilities = F.log_softmax(all_scores, dim=-1)

        return all_scores

    def multi_decode(
        self,
        og_string: torch.Tensor,
        input: torch.Tensor,
    ) -> List[List[str]]:
        """
        arguments
        ~~~~~~~~~
            og_string: original string tensor, assumes shape (batch x seq)
            input: input tensor with token types, assumes shape (batch x seq)
        """
        batch_size, seq_len = input.shape
        result = [
            [] for _ in range(batch_size)
        ]  # Create a list of lists for the result
        for b in range(batch_size):
            for i in range(seq_len):
                token = input[b, i].item()
                if token >= self.old_vocab_size + self.amount_of_relations:
                    # Copy token from the original string
                    copy_id = token - self.old_vocab_size - self.amount_of_relations
                    result[b].append(
                        self.tokenizer.convert_ids_to_tokens(
                            og_string[b, copy_id].item()
                        )
                    )
                elif token >= self.old_vocab_size:
                    # Map the relationship token to its string representation
                    rel_idx = token - self.tokenizer.vocab_size
                    str_rel = self.rel_list[rel_idx]  # type: ignore
                    result[b].append(str_rel)
                else:  # Normal token
                    # Decode the normal token
                    decoded_token = self.tokenizer.convert_ids_to_tokens(
                        input[b, i].item()
                    )
                    result[b].append(decoded_token)
        return result

    def _beamsearch(
        self,
        inputs: torch.Tensor,
        initial_states: torch.Tensor,
    ):
        with torch.no_grad():
            # We keep topk paths as such:
            batch_size = initial_states.shape[0]
            pad_token = self.tokenizer.pad_token_id
            prob_state_space_size = (
                self.tokenizer.vocab_size
                + self.amount_of_relations
                + self.bart_config.max_position_embeddings
            )

            # Input Stuff
            encoder_attn_mask = torch.ones_like(inputs)
            encoder_attn_mask[inputs == pad_token] = 0
            encoder_output = self.base.encoder(inputs, encoder_attn_mask)  # type: ignore
            encoder_states = encoder_output.last_hidden_state

            # (batch_size) x (beam_width) x (cur_seq_length)
            cur_sequences = initial_states.unsqueeze(-1)
            # OPTIM: we could store the decoder states
            probs_so_far = torch.zeros(
                (cur_sequences.size(0), cur_sequences.size(1)),
                device=initial_states.device,
            )

            stopping_criteria = torch.zeros(
                (batch_size, self.beam_width),
                device=initial_states.device,
                dtype=torch.int,
            )

            for s in range(40):  # type: ignore
                self.my_logger.debug(
                    f"Decoding {s}nth step cur_sequences shape : {cur_sequences.shape}. Possible decodings for input '{self.tokenizer.decode(inputs[0,:])}' are:"
                )
                for i in range(cur_sequences.size(1)):
                    self.my_logger.debug(
                        f"So far the decoding is: {self.tokenizer.decode(cur_sequences[0,i,:].squeeze())}"
                    )

                cur_beam_width = cur_sequences.size(1)
                ########################################
                # Organize and get probabilities of current choices
                ########################################

                cs_tensor = cur_sequences.view(batch_size * cur_beam_width, -1)
                rptd_enc_states = encoder_states.repeat_interleave(
                    cur_beam_width, dim=0
                )
                rptd_enc_attn = encoder_attn_mask.repeat_interleave(
                    cur_beam_width, dim=0
                )

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
                mixed_logits = self._mixed_logits(
                    rptd_enc_states, decoder_last_hidd_state
                )
                mixed_probabilities = F.softmax(mixed_logits, dim=-1)

                cum_probs = probs_so_far.view(batch_size * cur_beam_width, -1, 1) + (
                    1 * mixed_probabilities.log()
                )

                # Reconstruct to do topk across beams in single batch
                proper_probs = cum_probs.view(batch_size, -1)

                top_p, top_i = torch.topk(proper_probs, self.beam_width)
                new_past_idx = torch.Tensor(top_i // prob_state_space_size)
                top_i = (
                    top_i % prob_state_space_size
                ).view(  # Because beams get squished to batch dim
                    batch_size, self.beam_width, 1
                )

                ## Fix Copy:
                copy_mask = (
                    top_i >= self.tokenizer.vocab_size + self.amount_of_relations
                )  # .nonzero(as_tuple=False)
                ids = top_i[copy_mask] % (
                    self.tokenizer.vocab_size + self.amount_of_relations
                )
                batch_idx = copy_mask.nonzero(as_tuple=True)
                if len(batch_idx[0]) != 0:  # If there are ids to replace
                    enc_input_vals = inputs[batch_idx[0], ids]
                    top_i[copy_mask] = enc_input_vals

                # cur_sate is (batch_size x beam_width ) x (sequence length) in shape
                # probabilities are (batch_size x beam_width) x (beam_width)
                # We want a cartesian product  only of matching indices on (batch_size x beam_width)
                if cur_sequences.size(1) != self.beam_width:
                    cur_sequences = cur_sequences.repeat_interleave(
                        self.beam_width, dim=1
                    )
                    probs_so_far = probs_so_far.repeat_interleave(
                        self.beam_width, dim=1
                    )
                probs_so_far += top_p

                # Reselect history based on top_i
                for b in range(batch_size):  # OPTIM: See if you can remove loop
                    cur_sequences[b, :, :] = torch.index_select(
                        cur_sequences[b], 0, new_past_idx[b]
                    )  # type:ignore
                cur_sequences = torch.cat((cur_sequences, top_i), dim=-1)

                # </s> token present
                fwdss_id = self.tokenizer.convert_tokens_to_ids(["</s>"])[0]
                # Boolean or on Tensors
                stopping_criteria = stopping_criteria | (top_i.squeeze(-1) == fwdss_id)

                if torch.all(stopping_criteria):
                    break

            self.my_logger.info(
                f"Returning cur_sequences with shape {cur_sequences.shape}"
            )
            return cur_sequences

    def _teacher_forcing_inputs(
        self,
        encoder_input: torch.Tensor,
        hybrid_targets: torch.Tensor,
        token_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_size = self.tokenizer.vocab_size
        amnt_rels = self.amount_of_relations
        # Change Values
        mask_rel = token_types == TokenType.RELATIONSHIP.value
        mask_copy = token_types == TokenType.COPY.value
        vocab_target = hybrid_targets.clone()
        expandvocab_target = hybrid_targets.clone()

        # Replace
        for i in range(hybrid_targets.size(0)):
            copy_idxs = torch.where(mask_copy[i, :])[0]
            encoder_seq_idx = hybrid_targets[i, copy_idxs]
            encoder_vals = encoder_input[i, encoder_seq_idx]
            vocab_target[i, copy_idxs] = encoder_vals

        vocab_target[mask_rel] += vocab_size

        expandvocab_target[mask_rel] += vocab_size
        expandvocab_target[mask_copy] += vocab_size + amnt_rels

        return vocab_target, expandvocab_target

    def training_step(self, batches, batch_idx):
        assert isinstance(self.base, BartModel)
        # self.my_logger.debug(f"Going through batch idx {batch_idx}")
        inputs, hybrid_target, token_types = batches
        padding_token = self.tokenizer.pad_token_id

        # Create Teacher Forcing Decoder Inputs
        copy_fixed_target, expand_vocab_target = self._teacher_forcing_inputs(
            inputs, hybrid_target, token_types
        )
        # TODO: replace `target` with ready `vocab_target`

        # Generates Masks
        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == padding_token] = 0
        encoder_outputs = self.base.encoder(inputs, encoder_attn_mask).last_hidden_state

        decoder_attn_mask = torch.ones_like(copy_fixed_target)
        decoder_attn_mask[copy_fixed_target == padding_token] = 0
        decoder_hidden_outputs = self.base.decoder(
            copy_fixed_target, decoder_attn_mask, encoder_outputs, encoder_attn_mask
        ).last_hidden_state

        mixed_logits = self._mixed_logits(encoder_outputs, decoder_hidden_outputs)
        mixed_logsofts = F.log_softmax(mixed_logits, dim=-1)

        flat_probabilities = mixed_logsofts.view(-1, mixed_logsofts.size(-1))

        # Hybrid T
        flat_target = expand_vocab_target.view(-1)

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

        copy_fixed_target, expand_vocab_target = self._teacher_forcing_inputs(
            inputs, target, token_types
        )

        encoder_attn_mask = torch.ones_like(inputs)
        encoder_attn_mask[inputs == padding_token] = 0
        encoder_outputs = self.base.encoder(inputs, encoder_attn_mask).last_hidden_state

        decoder_attn_mask = torch.ones_like(copy_fixed_target)
        decoder_attn_mask[copy_fixed_target == padding_token] = 0
        decoder_hidden_outputs = self.base.decoder(
            copy_fixed_target, decoder_attn_mask, encoder_outputs, encoder_attn_mask
        ).last_hidden_state

        mixed_logits = self._mixed_logits(encoder_outputs, decoder_hidden_outputs)
        mixed_logsofts = F.log_softmax(mixed_logits, dim=-1)

        flat_probabilities = mixed_logsofts.view(-1, mixed_logsofts.size(-1))
        flat_target = expand_vocab_target.view(-1)

        # loss = self.criterion(mixed_probabilities, target, masks)
        loss = self.criterion(flat_probabilities, flat_target)
        loss_avg = loss.mean()
        self.log(
            "val_loss", loss_avg.item(), prog_bar=True, on_step=True, on_epoch=True
        )
        if batch_idx == 0:
            # Given a source string of :
            dict_to_log = {}
            dict_to_log[
                "val_src_txt"
            ] = f"Source text is {str(self.tokenizer.decode(inputs[0]).split(' '))}"

            trans_target = self.multi_decode(
                inputs[0].unsqueeze(0), expand_vocab_target[0].unsqueeze(0)
            )
            dict_to_log["val_target"] = f"Target is {str(target[0])}"
            dict_to_log["val_transtarget"] = f"TransTarget is {str(trans_target)}"

            actual_probs = F.softmax(mixed_logits[0], dim=-1)
            estimated_tokens = torch.argmax(actual_probs, dim=-1)
            translated_estimation = self.multi_decode(
                inputs[0].unsqueeze(0), estimated_tokens.unsqueeze(0)
            )
            dict_to_log[
                "val_est_tok"
            ] = f"Estimated (expanded) ids {str(estimated_tokens)}"
            dict_to_log[
                "val_trans_esttok"
            ] = f"Translated Estimation is {str(translated_estimation)}"
            # self.logger.experiment.log({"val_report": dict_to_log})
            wdb_table = wandb.Table(
                columns=list(dict_to_log.keys()),
                data=list(dict_to_log.values()),
            )
            self.loggger.log({"val_report": wdb_table})
            # wandb.log({"val_report": dict_to_log})

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

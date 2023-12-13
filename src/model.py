# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import logging
import types

import numpy as np
import torch
import torch.nn.functional as F
import transformers
import wandb
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5Block,
    T5LayerCrossAttention,
    T5LayerNorm,
    T5LayerSelfAttention,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map


def get_key_value_states_only(
    self,
    hidden_states,
    key_value_states=None,
    past_key_value=None,
    query_length=None,
):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
            len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get key/value states
    key_states = project(
        hidden_states,
        self.k,
        key_value_states,
        past_key_value[0] if past_key_value is not None else None,
    )
    value_states = project(
        hidden_states,
        self.v,
        key_value_states,
        past_key_value[1] if past_key_value is not None else None,
    )

    return key_states, value_states


def get_self_attn_key_value(
    self,
    hidden_states,
    past_key_value,
):
    """
    hidden_states,
    key_value_states=None,
    past_key_value=None,
    query_length=None,
    """
    normed_hidden_states = self.layer_norm(hidden_states)
    return self.SelfAttention.get_key_value_states_only(
        normed_hidden_states,
        past_key_value=past_key_value,
    )


def get_cross_attn_key_value(
    self,
    hidden_states,
    key_value_states,
    past_key_value,
    query_length,
):
    """
    hidden_states,
    key_value_states=None,
    past_key_value=None,
    query_length=None,
    """
    normed_hidden_states = self.layer_norm(hidden_states)
    return self.EncDecAttention.get_key_value_states_only(
        normed_hidden_states,
        key_value_states=key_value_states,
        past_key_value=past_key_value,
        query_length=query_length,
    )


def get_block_project_key_value(
    self,
    hidden_states,
    encoder_hidden_states=None,
    past_key_value=None,
):
    if past_key_value is not None:
        assert self.is_decoder, "Only decoder can use `past_key_values`"
        expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

        if len(past_key_value) != expected_num_past_key_values:
            raise ValueError(
                f"There should be {expected_num_past_key_values} past states. "
                f"{'2 (past / key) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                f"Got {len(past_key_value)} past key / value states"
            )

        self_attn_past_key_value = past_key_value[:2]
        cross_attn_past_key_value = past_key_value[2:]
    else:
        self_attn_past_key_value, cross_attn_past_key_value = None, None

    present_key_value_state = self.layer[0].get_self_attn_key_value(
        hidden_states,
        past_key_value=self_attn_past_key_value,
    )

    # NOTE: the cross attention compute is based on the hidden states from the self attention
    # Here, we feed the same hidden sates to each layer independantly, knowing that it could
    # introduce an error
    # hidden_states, present_key_value_state = self_attention_outputs[:2]

    # the actual query length is unknown for cross attention
    # if using past key value states. Need to inject it here
    if present_key_value_state is not None:
        query_length = present_key_value_state[0].shape[2]
    else:
        query_length = None

    cross_present_key_value_state = self.layer[1].get_cross_attn_key_value(
        hidden_states,
        key_value_states=encoder_hidden_states,
        past_key_value=cross_attn_past_key_value,
        query_length=query_length,
    )

    # Combine self attn and cross attn key value states
    present_key_value_state = present_key_value_state + cross_present_key_value_state

    return present_key_value_state


T5Attention.get_key_value_states_only = get_key_value_states_only
T5LayerSelfAttention.get_self_attn_key_value = get_self_attn_key_value
T5LayerCrossAttention.get_cross_attn_key_value = get_cross_attn_key_value
T5Block.get_block_project_key_value = get_block_project_key_value


def get_lambda_at_token(lamb, t, N, tau, alpha):
    return np.clip(alpha * lamb + (1 - alpha) * np.exp(-t * tau / N), a_min=0, a_max=1)


def wandb_log(d):
    if wandb.run is not None:
        wandb.log(d)


def get_distil_loss(p, q, T=1):
    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    inp = F.log_softmax(q / T, dim=-1)
    target = F.softmax(p / T, dim=-1)
    output = (T**2) * kl_loss(inp, target)
    return output


def get_layer_loss_weights(num_decoder_layers):
    layer_index_list = np.array(range(1, num_decoder_layers + 1))
    layer_index_list_sum = np.sum(layer_index_list)
    layer_index_list_weights = layer_index_list / layer_index_list_sum
    return layer_index_list_weights


class DecoderLayerHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.config = config
        self.model_dim = config.d_model

    def forward(self, hidden_states):
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)

        lm_logits = self.lm_head(hidden_states)
        return lm_logits


class MLPHead(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = torch.nn.Linear(config.d_model, config.d_model)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.out_proj = torch.nn.Linear(config.d_model, 1)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


CLASSIFIER_TYPES = {
    "2layer": lambda config: MLPHead(config),
    "simple": lambda config: torch.nn.Linear(config.d_model, 1),
}


class ConfidenceHeads(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.share_conf_heads:
            self.head = CLASSIFIER_TYPES[config.conf_head_type](config)
        else:
            self.heads = torch.nn.ModuleList(
                [
                    CLASSIFIER_TYPES[config.conf_head_type](config)
                    for _ in range(config.num_decoder_layers - 1)
                ]
            )

    def forward(self, x, i):
        if self.config.share_conf_heads:
            return self.head(x)
        else:
            return self.heads[i](x)


def get_scores(attn_list, attn_mask, n_passages, config):
    scores = torch.stack(attn_list)
    return scores


def get_passage_scores(
    all_cross_attentions_base, present_key_value_states_base, mask, config, passage_count=100
):
    all_normalized_attn_scores = []
    if config.filter_use_last_state:
        all_cross_attentions = [all_cross_attentions_base[-1]]
        present_key_value_states = [present_key_value_states_base[-1]]
    else:
        all_cross_attentions = all_cross_attentions_base
        present_key_value_states = present_key_value_states_base

    for ca_cur, pkv_cur in zip(all_cross_attentions, present_key_value_states):
        if config.filter_use_values:
            # pkv = (self_k, self_v, cross_k, cross_v)
            # note that the values can be early exited
            mean_attn_scores = (torch.norm(pkv_cur[3].float(), dim=-1)[:, :, None] * ca_cur).mean(
                dim=1
            )
        else:
            mean_attn_scores = ca_cur.mean(dim=1)
        all_normalized_attn_scores.append(mean_attn_scores)

    scores_filled = get_scores(all_normalized_attn_scores, mask, passage_count, config)

    scores_filled = scores_filled.sum(dim=[0, 1, 2])
    return scores_filled


def filter_past_key_value_states(pkv, passage_mask, config):
    return pkv[:, :, passage_mask, :]


class T5Stack(transformers.T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        self.block = nn.ModuleList(
            [
                T5Block(config, has_relative_attention_bias=bool(i == 0))
                for i in range(config.num_layers)
            ]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        if self.is_decoder:
            if hasattr(self.config, "train_conf_heads") and self.config.train_conf_heads:
                self.confidence_head_estimator = ConfidenceHeads(config)

            self.use_shared_decoder_lm_head = (
                hasattr(config, "use_shared_decoder_lm_head") and config.use_shared_decoder_lm_head
            )

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        self.ee_layers = []
        self.filter_hidden_states = False

    def parallelize(self, device_map=None):
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        )
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        # Load onto devices
        for k, v in self.device_map.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def get_softmax_diff(self, h, layer_index):
        if self.use_shared_decoder_lm_head:
            pred = self.get_lm_logits_output(h)

        pred_dist = torch.softmax(pred, dim=-1)
        pred_dist_top2 = pred_dist.topk(2, dim=-1)
        top_2_diff = (pred_dist_top2.values[:, :, 0] - pred_dist_top2.values[:, :, 1]).abs().min()
        return pred, top_2_diff

    def get_state_diff(self, h, prev_h):
        cos = torch.nn.CosineSimilarity(dim=-1)
        output = cos(h, prev_h).abs().min()
        return output

    def get_threshold_at_token(self, token):
        if self.config.decoder_early_exit_tau == -1:
            return self.config.decoder_early_exit_thres

        return get_lambda_at_token(
            lamb=self.config.decoder_early_exit_thres,
            t=token - 1,
            N=self.config.answer_maxlength,
            tau=self.config.decoder_early_exit_tau,
            alpha=self.config.decoder_early_exit_alpha,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)

        if self.filter_hidden_states:
            passage_mask = self.decoder_passage_mask
            encoder_hidden_states = encoder_hidden_states[:, passage_mask, :]

            if encoder_attention_mask is not None:
                encoder_attention_mask = encoder_attention_mask[:, passage_mask]

        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape, inputs_embeds.device
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        compute_hidden_states = True
        prev_hidden_states = None
        exit_layer_index = -1
        precomputed_lm_head_output = None
        output_attention_for_decoder_pass_filter = False

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            if (
                self.filter_hidden_states
                and past_key_value is not None
                and past_key_value[2].shape[2] > encoder_hidden_states.shape[1]
            ):
                filtered_pk = filter_past_key_value_states(
                    past_key_value[2], passage_mask, self.config
                )
                filtered_pv = filter_past_key_value_states(
                    past_key_value[3], passage_mask, self.config
                )
                past_key_value = (past_key_value[0], past_key_value[1], filtered_pk, filtered_pv)

            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                        hidden_states.device
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.is_decoder and self.config.filter:
                current_token = 0 if past_key_value is None else past_key_value[0].shape[2]
                output_attention_for_decoder_pass_filter = current_token == self.config.filter_token
                output_attentions = output_attentions or output_attention_for_decoder_pass_filter
                if all_attentions is None:
                    all_attentions = ()
                    all_cross_attentions = ()

            if compute_hidden_states:
                if self.gradient_checkpointing and self.training:
                    if use_cache:
                        logging.warn(
                            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                        )
                        use_cache = False

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return tuple(module(*inputs, use_cache, output_attentions))

                        return custom_forward

                    layer_outputs = checkpoint(
                        create_custom_forward(layer_module),
                        hidden_states,
                        extended_attention_mask,
                        position_bias,
                        encoder_hidden_states,
                        encoder_extended_attention_mask,
                        encoder_decoder_position_bias,
                        layer_head_mask,
                        cross_attn_layer_head_mask,
                        None,  # past_key_value is always None with gradient checkpointing
                    )
                else:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_extended_attention_mask,
                        encoder_decoder_position_bias=encoder_decoder_position_bias,
                        layer_head_mask=layer_head_mask,
                        cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                        past_key_value=past_key_value,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                    )
            else:
                present_key_value_state = layer_module.get_block_project_key_value(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_value=past_key_value,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            if compute_hidden_states:
                hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

            if output_attention_for_decoder_pass_filter and i == self.config.filter_layer:
                ca_passage_scores = get_passage_scores(
                    all_cross_attentions,
                    present_key_value_states,
                    encoder_extended_attention_mask,
                    self.config,
                    self.config.n_context,
                )
                ca_passage_ranks = (
                    ca_passage_scores.view(-1).argsort(descending=True).argsort(descending=False)
                )

                tokens_to_take = ca_passage_ranks.shape[0] * self.config.filter_to_take_percent
                ca_passage_ranks_mask = ca_passage_ranks < tokens_to_take
                self.decoder_passage_mask = ca_passage_ranks_mask

            if (
                self.is_decoder
                and self.config.decoder_early_exit_type is not None
                and compute_hidden_states
            ):
                if past_key_value is None:
                    decoder_early_exit_thres = self.config.decoder_early_exit_thres
                else:
                    decoder_early_exit_thres = self.get_threshold_at_token(
                        past_key_value[0].shape[2]
                    )

                if (
                    self.config.decoder_early_exit_type == "softmax"
                    and i < self.config.num_decoder_layers - 1
                ):
                    precomp_lm, ee_diff = self.get_softmax_diff(hidden_states, i)
                    if ee_diff > decoder_early_exit_thres:
                        compute_hidden_states = False
                        exit_layer_index = i
                        precomputed_lm_head_output = precomp_lm

                if self.config.decoder_early_exit_type == "state":
                    if prev_hidden_states is not None:
                        ee_diff = self.get_state_diff(hidden_states, prev_hidden_states)
                        if ee_diff > decoder_early_exit_thres:
                            compute_hidden_states = False
                            exit_layer_index = i

                    prev_hidden_states = hidden_states

                if (
                    self.config.decoder_early_exit_type == "classifier"
                    and i < self.config.num_decoder_layers - 1
                ):
                    # NOTICE: applying sigmoid due to loss being BCE Loss
                    current_confidence_estimation = torch.sigmoid(
                        self.confidence_head_estimator(hidden_states, i)
                    )
                    current_confidence_estimation = current_confidence_estimation.min()
                    if current_confidence_estimation > decoder_early_exit_thres:
                        compute_hidden_states = False
                        exit_layer_index = i

                if not compute_hidden_states:
                    self.ee_layers.append(i)

        if compute_hidden_states:
            hidden_states = self.final_layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

        if output_attention_for_decoder_pass_filter:
            self.filter_hidden_states = True

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    exit_layer_index,
                    precomputed_lm_head_output,
                ]
                if v is not None
            )
        model_output_ca = BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )
        model_output_ca.exit_layer_index = exit_layer_index
        model_output_ca.precomputed_lm_head_output = precomputed_lm_head_output
        return model_output_ca


class T5ForConditionalGeneration(transformers.T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.use_shared_decoder_lm_head = (
            hasattr(config, "use_shared_decoder_lm_head") and config.use_shared_decoder_lm_head
        )
        self.train_shared_decoder_lm_head = self.use_shared_decoder_lm_head
        self.train_last_lm_layer = True
        self.perform_train_conf_heads = False

        if hasattr(config, "train_conf_heads") and config.train_conf_heads:
            self.train_shared_decoder_lm_head = False
            self.train_last_lm_layer = False
            self.perform_train_conf_heads = True

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        setting="train",
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states or self.train_shared_decoder_lm_head,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if decoder_outputs.exit_layer_index == -1:
            # Set device for model parallelism
            if self.model_parallel:
                torch.cuda.set_device(self.encoder.first_device)
                self.lm_head = self.lm_head.to(self.encoder.first_device)
                sequence_output = sequence_output.to(self.lm_head.weight.device)

            if self.config.tie_word_embeddings:
                # Rescale output before projecting on vocab
                # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
                sequence_output = sequence_output * (self.model_dim**-0.5)

            lm_logits = self.lm_head(sequence_output)
        else:
            if decoder_outputs.precomputed_lm_head_output is None:
                if self.use_shared_decoder_lm_head:
                    lm_logits = self.get_lm_logits_output(sequence_output)
            else:
                lm_logits = decoder_outputs.precomputed_lm_head_output

        loss = None
        if labels is not None:
            loss_weights_log = {}
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            if self.train_last_lm_layer:
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

                loss_weights_log[f"{setting}_final_decoder_layer_loss"] = loss.item()

            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            if self.train_shared_decoder_lm_head:
                loss_weights = get_layer_loss_weights(self.config.num_decoder_layers)

                loss = loss_weights[-1] * loss

                for ee_index in range(self.config.num_decoder_layers - 1):
                    current_decoder_states = decoder_outputs.hidden_states[ee_index + 1]

                    current_lm_logits = self.get_lm_logits_output(current_decoder_states)

                    current_decoder_loss = loss_fct(
                        current_lm_logits.view(-1, current_lm_logits.size(-1)), labels.view(-1)
                    )
                    loss_weights_log[
                        f"{setting}_decoder_layer_loss_{ee_index}"
                    ] = current_decoder_loss.item()
                    loss = loss + loss_weights[ee_index] * current_decoder_loss

            if len(loss_weights_log) > 0:
                wandb_log(loss_weights_log)

            if self.perform_train_conf_heads:
                pred_L_argmax = lm_logits.argmax(dim=-1)
                conf_loss_log = {}
                loss = 0

                for ee_index in range(self.config.num_decoder_layers - 1):
                    current_decoder_states = decoder_outputs.hidden_states[ee_index + 1]

                    cur_conf_loss, conf_metrics = self.get_confidence_loss(
                        current_decoder_states, ee_index, pred_L_argmax, setting
                    )

                    conf_loss_log[f"{setting}_decoder_conf_loss_{ee_index}"] = cur_conf_loss.item()
                    loss = loss + cur_conf_loss
                    conf_loss_log.update(conf_metrics)

                wandb_log(conf_loss_log)

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def get_confidence_loss(self, hidden_states, i, pred_L_argmax, setting):
        pred_i = self.get_lm_logits_output(hidden_states)

        pred_i_argmax = pred_i.argmax(dim=-1)

        pred_oracle = (pred_i_argmax == pred_L_argmax).to(hidden_states.dtype).detach()

        confidence_scores = self.decoder.confidence_head_estimator(hidden_states, i)

        loss_fn = torch.nn.BCEWithLogitsLoss()

        conf_loss = loss_fn(confidence_scores.view(*pred_oracle.shape), pred_oracle)

        conf_metrics = {}
        with torch.no_grad():
            precision, recall, _ = precision_recall_curve(
                y_true=pred_oracle.view(-1).tolist(),
                probas_pred=confidence_scores.view(-1).tolist(),
            )
            auc_precision_recall = auc(recall, precision)
            conf_metrics[f"{setting}_auc_precision_recall_{i}"] = auc_precision_recall

        return conf_loss, conf_metrics

    def get_lm_logits_output(self, hidden_states):
        hidden_states = self.decoder.final_layer_norm(hidden_states)
        hidden_states = self.decoder.dropout(hidden_states)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            hidden_states = hidden_states * (self.model_dim**-0.5)

        lm_logits = self.lm_head(hidden_states)
        return lm_logits

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past


class FiDT5(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()

    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(kwargs["input_ids"].size(0), -1)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(
                kwargs["attention_mask"].size(0), -1
            )

        return super(FiDT5, self).forward(**kwargs)

    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        return super().forward(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=return_dict, **kwargs
        )

    # We need to resize the inputs here, as the generate method expect 2D tensors
    def generate(
        self,
        input_ids,
        attention_mask,
        max_length,
        min_length,
        num_beams,
        no_repeat_ngram_size,
        early_stopping,
    ):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=early_stopping,
        )

    def generate_short(self, input_ids, attention_mask, max_length):
        self.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids=input_ids.view(input_ids.size(0), -1),
            attention_mask=attention_mask.view(attention_mask.size(0), -1),
            max_length=max_length,
        )

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)
        self.encoder.main_input_name = self.main_input_name

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, state_dict):
        self.unwrap_encoder()

        current_state_dict = self.state_dict()

        missing_keys = set(current_state_dict.keys()).symmetric_difference(set(state_dict.keys()))

        for k in missing_keys:
            logging.info(f"fix missing {k}")
            state_dict[k] = current_state_dict[k]

        self.load_state_dict(state_dict)
        self.wrap_encoder()

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = context_mask.size(1)
        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage)
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, _ = scores.size()
        # batch_size, n_head, n_layers, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, n_passages, -1)
        scores = scores.masked_fill(~context_mask[:, None, None], 0.0)
        scores = scores.sum(dim=[1, 2, 4])
        ntokens = context_mask.sum(dim=[2]) * n_layers * n_heads
        scores = scores / ntokens
        return scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """

    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()

        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        return_dict=True,
        **kwargs,
    ):
        # total_length = n_passages * passage_length
        bsz, total_length = input_ids.shape
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz * self.n_passages, passage_length)
        attention_mask = attention_mask.view(bsz * self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, return_dict=return_dict, **kwargs)
        outputs.last_hidden_state = outputs.last_hidden_state.view(
            bsz, self.n_passages * passage_length, -1
        )
        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """

    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [], dtype=torch.float, device=output[0].device, requires_grad=True
                )
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward, hidden_states, attention_mask, position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
    self,
    input,
    mask=None,
    kv=None,
    position_bias=None,
    past_key_value_state=None,
    head_mask=None,
    query_length=None,
    use_cache=False,
    output_attentions=False,
):
    """
    This only works for computing cross attention over the input
    """
    assert kv != None
    assert head_mask == None
    assert position_bias != None or self.has_relative_attention_bias

    bsz, qlen, dim = input.size()
    n_heads, d_heads = self.n_heads, self.d_kv
    klen = kv.size(1)

    q = self.q(input).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    if past_key_value_state == None:
        k = self.k(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
        v = self.v(kv).view(bsz, -1, n_heads, d_heads).transpose(1, 2)
    else:
        k, v = past_key_value_state

    scores = torch.einsum("bnqd,bnkd->bnqk", q, k)

    if mask is not None:
        scores += mask

    if position_bias is None:
        position_bias = self.compute_bias(qlen, klen)
    scores += position_bias

    if self.score_storage is None:
        self.score_storage = scores

    attn = F.softmax(scores.float(), dim=-1).type_as(scores)
    attn = F.dropout(attn, p=self.dropout, training=self.training)

    output = torch.matmul(attn, v)
    output = output.transpose(1, 2).contiguous().view(bsz, -1, self.inner_dim)
    output = self.o(output)

    if use_cache:
        output = (output,) + ((k, v),)
    else:
        output = (output,) + (None,)

    if output_attentions:
        output = output + (attn,)

    if self.has_relative_attention_bias:
        output = output + (position_bias,)

    return output


class RetrieverConfig(transformers.BertConfig):
    def __init__(
        self,
        indexing_dimension=768,
        apply_question_mask=False,
        apply_passage_mask=False,
        extract_cls=False,
        passage_maxlength=200,
        question_maxlength=40,
        projection=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.indexing_dimension = indexing_dimension
        self.apply_question_mask = apply_question_mask
        self.apply_passage_mask = apply_passage_mask
        self.extract_cls = extract_cls
        self.passage_maxlength = passage_maxlength
        self.question_maxlength = question_maxlength
        self.projection = projection


class Retriever(transformers.PreTrainedModel):
    config_class = RetrieverConfig
    base_model_prefix = "retriever"

    def __init__(self, config, initialize_wBERT=False):
        super().__init__(config)
        assert (
            config.projection or config.indexing_dimension == 768
        ), "If no projection then indexing dimension must be equal to 768"
        self.config = config
        if initialize_wBERT:
            self.model = transformers.BertModel.from_pretrained("bert-base-uncased")
        else:
            self.model = transformers.BertModel(config)
        if self.config.projection:
            self.proj = nn.Linear(self.model.config.hidden_size, self.config.indexing_dimension)
            self.norm = nn.LayerNorm(self.config.indexing_dimension)
        self.loss_fct = torch.nn.KLDivLoss()

    def forward(self, question_ids, question_mask, passage_ids, passage_mask, gold_score=None):
        question_output = self.embed_text(
            text_ids=question_ids,
            text_mask=question_mask,
            apply_mask=self.config.apply_question_mask,
            extract_cls=self.config.extract_cls,
        )
        bsz, n_passages, plen = passage_ids.size()
        passage_ids = passage_ids.view(bsz * n_passages, plen)
        passage_mask = passage_mask.view(bsz * n_passages, plen)
        passage_output = self.embed_text(
            text_ids=passage_ids,
            text_mask=passage_mask,
            apply_mask=self.config.apply_passage_mask,
            extract_cls=self.config.extract_cls,
        )

        score = torch.einsum(
            "bd,bid->bi", question_output, passage_output.view(bsz, n_passages, -1)
        )
        score = score / np.sqrt(question_output.size(-1))
        if gold_score is not None:
            loss = self.kldivloss(score, gold_score)
        else:
            loss = None

        return question_output, passage_output, score, loss

    def embed_text(self, text_ids, text_mask, apply_mask=False, extract_cls=False):
        text_output = self.model(
            input_ids=text_ids, attention_mask=text_mask if apply_mask else None
        )
        if type(text_output) is not tuple:
            text_output.to_tuple()
        text_output = text_output[0]
        if self.config.projection:
            text_output = self.proj(text_output)
            text_output = self.norm(text_output)

        if extract_cls:
            text_output = text_output[:, 0]
        else:
            if apply_mask:
                text_output = text_output.masked_fill(~text_mask[:, :, None], 0.0)
                text_output = torch.sum(text_output, dim=1) / torch.sum(text_mask, dim=1)[:, None]
            else:
                text_output = torch.mean(text_output, dim=1)
        return text_output

    def kldivloss(self, score, gold_score):
        gold_score = torch.softmax(gold_score, dim=-1)
        score = torch.nn.functional.log_softmax(score, dim=-1)
        return self.loss_fct(score, gold_score)

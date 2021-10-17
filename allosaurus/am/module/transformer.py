import logging
import torch
import torch.nn as nn
from allosaurus.am.module.ffn import PositionwiseFeedForward
from allosaurus.am.module.attention import MultiHeadedSelfAttention, MultiHeadedSelfAttentionWithRelPos

logger = logging.getLogger(__name__)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()

        self.relative_positional = config.relative_positional

        if self.relative_positional:
            self.slf_attn = MultiHeadedSelfAttentionWithRelPos(config.head_size, config.hidden_size,
                                                               config.slf_attn_dropout)
        else:
            self.slf_attn = MultiHeadedSelfAttention(config.head_size, config.hidden_size, config.slf_attn_dropout)
        self.feed_forward = PositionwiseFeedForward(config.hidden_size, config.d_ff, config.ffn_dropout,
                                                    config.activation)

        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)

        self.dropout1 = nn.Dropout(config.residual_dropout)
        self.dropout2 = nn.Dropout(config.residual_dropout)

        self.normalize_before = config.normalize_before
        self.concat_after = config.concat_after

        if self.concat_after:
            self.concat_linear = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, x, mask, pos=None):

        if self.normalize_before:
            x = self.norm1(x)
        residual = x

        if self.relative_positional:
            slf_attn_out, slf_attn_weights = self.slf_attn(x, mask, pos)
        else:
            slf_attn_out, slf_attn_weights = self.slf_attn(x, mask)

        if self.concat_after:
            x = residual + self.concat_linear(torch.cat((x, slf_attn_out), dim=-1))
        else:
            x = residual + self.dropout1(slf_attn_out)
        if not self.normalize_before:
            x = self.norm1(x)

        if self.normalize_before:
            x = self.norm2(x)
        residual = x
        x = residual + self.dropout2(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, {'slf_attn_weights': slf_attn_weights}

    def inference(self, x, mask, pos=None, cache=None):
        if self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.relative_positional:
            slf_attn_out, slf_attn_weights, new_cache = self.slf_attn.inference(x, mask, cache, pos)
        else:
            slf_attn_out, slf_attn_weights, new_cache = self.slf_attn.inference(x, mask, cache)

        if self.concat_after:
            x = residual + self.concat_linear(torch.cat((x, slf_attn_out), dim=-1))
        else:
            x = residual + slf_attn_out
        if not self.normalize_before:
            x = self.norm1(x)

        if self.normalize_before:
            x = self.norm2(x)
        residual = x
        x = residual + self.feed_forward(x)
        if not self.normalize_before:
            x = self.norm2(x)

        return x, new_cache, {'slf_attn_weights': slf_attn_weights}


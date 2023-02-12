# File   : attention.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import math
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BasedAttention(nn.Module):
    def __init__(self, source_dim, output_dim, enable_output_proj=True, dropout=0.0):
        super(BasedAttention, self).__init__()

        self.enable_output_proj = enable_output_proj
        if self.enable_output_proj:
            self.output_proj = nn.Linear(source_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def compute_context(self, values, scores, mask=None):
        """
        Args:
            values: [b, t2, v] or [b, nh, t2, v]
            scores: [b, t1, t2] or [b, nh, t1, t2]
            mask: [b, t1, t2] or [b, 1/nh, t1, t2]
        """

        assert values.dim() == scores.dim()

        if mask is not None:
            scores.masked_fill_(~mask, -float('inf'))
        
        weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(weights, values)

        if context.dim() == 4:
            b, n, t, v = context.size()
            context = context.transpose(1, 2).reshape(b, t, n * v)
        
        if self.enable_output_proj:
            context = self.output_proj(context)

        return self.dropout(context), weights


class MultiHeadedSelfAttention(BasedAttention):
    def __init__(self, head_size, hidden_size, dropout_rate=0.0, share_qvk_proj=False):
        super(MultiHeadedSelfAttention, self).__init__(hidden_size, hidden_size, enable_output_proj=True, dropout=dropout_rate)

        self.hidden_size = hidden_size
        self.share_qvk_proj = share_qvk_proj
        self.nheads = head_size
        self.d_k = hidden_size // head_size

        self.qvk_proj = nn.Linear(hidden_size, hidden_size if self.share_qvk_proj else hidden_size * 3)

    def forward(self, x, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, hidden_size)
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.hidden_size, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights

    def inference(self, x, mask, cache=None):

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.hidden_size, dim=-1)

        batch_size = x.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights, cache


class MultiHeadedCrossAttention(BasedAttention):
    def __init__(self, head_size, hidden_size, memory_dim, dropout_rate=0.0, share_vk_proj=False):
        super(MultiHeadedCrossAttention, self).__init__(hidden_size, hidden_size, enable_output_proj=True, dropout=dropout_rate)

        self.hidden_size = hidden_size
        self.share_vk_proj = share_vk_proj
        self.nheads = head_size
        self.d_k = hidden_size // head_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.vk_proj = nn.Linear(memory_dim, hidden_size if self.share_vk_proj else hidden_size * 2)

    def forward(self, query, memory, memory_mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, hidden_size)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = torch.split(memory, self.hidden_size, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights

    def inference(self, query, memory, memory_mask, cache=None):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor memory: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1 or 1, time2)
        :return torch.Tensor: attentined and transformed `value` (batch, time1, hidden_size)
        """

        query = self.q_proj(query)
        memory = self.vk_proj(memory)

        if self.share_vk_proj:
            key = value = memory
        else:
            key, value = torch.split(memory, self.hidden_size, dim=-1)

        batch_size = query.size(0)
        query = query.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(query, key.transpose(2, 3)) / math.sqrt(self.d_k)

        context, attn_weights = self.compute_context(value, scores, memory_mask.unsqueeze(1))

        return context, attn_weights, cache


class MultiHeadedSelfAttentionWithRelPos(BasedAttention):
    def __init__(self, head_size, hidden_size, dropout_rate=0.0, skip_term_b=False, share_qvk_proj=False):
        super(MultiHeadedSelfAttentionWithRelPos, self).__init__(head_size, hidden_size, dropout_rate, share_qvk_proj)

        self.hidden_size = hidden_size
        self.share_qvk_proj = share_qvk_proj
        self.skip_term_b = skip_term_b
        self.nheads = head_size
        self.d_k = hidden_size // head_size

        self.qvk_proj = nn.Linear(hidden_size, hidden_size if self.share_qvk_proj else hidden_size * 3)

        self.pos_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        self.posu = nn.Parameter(torch.Tensor(1, 1, head_size, self.d_k))
        self.posv = nn.Parameter(torch.Tensor(1, 1, head_size, self.d_k))

        torch.nn.init.xavier_normal_(self.posu)
        torch.nn.init.xavier_normal_(self.posv)

    def _RelPosBias(self, content, abs_pos):
        """Compute relative positinal encoding.
        Args:
            content: [B, T, N, H] if not self.skip_term_b else [1, 1, N, H]
            abs_pos: [B, N, S=2T-1, H]
        Returns:
            torch.Tensor: Output tensor.
        """
        B, _, N, _ = content.size()
        S= abs_pos.size(2)
        T = (S + 1) // 2

        if not self.skip_term_b:
            matrix_bd = torch.matmul(content.transpose(1, 2), abs_pos.transpose(-2, -1)) # [B, N, T, S]
        else:
            matrix_bd = torch.matmul(content.transpose(1, 2), abs_pos.transpose(-2, -1)) # [1, 1, T, S]

        rel_pos = torch.arange(0, T, dtype=torch.long, device=matrix_bd.device)
        rel_pos = (rel_pos[None] - rel_pos[:, None]).reshape(1, 1, T, T) + (T - 1)
        return torch.gather(matrix_bd, dim=3, index=rel_pos.repeat(B, N, 1, 1))

    def forward(self, x, mask, pos):
        """
        Args:
            x: [B, T, V]
            mask: [B, 1, T]
            pos: positional embedding [B, S=2T-1, V]
        """

        x = self.qvk_proj(x)

        if self.share_qvk_proj:
            query = key = value = x
        else:
            query, key, value = torch.split(x, self.hidden_size, dim=-1)

        batch_size = x.size(0)
        # [B, T, V] -> [B, T, N, H]
        query = query.reshape(batch_size, -1, self.nheads, self.d_k)
        # [B, T, V] -> [B, H, T, H]
        key = key.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)
        value = value.reshape(batch_size, -1, self.nheads, self.d_k).transpose(1, 2)

        bpos = pos.size(0)
        # [B, S, V] -> [B, S, N, H] -> [B, N, S, H]
        pos = self.pos_proj(pos).reshape(bpos, -1, self.nheads, self.d_k).transpose(1, 2)

        # [B, T, N, H] = [B, T, N, H] + [1, 1, N, H]
        query_with_bias_u = query + self.posu
        query_with_bias_u = query_with_bias_u.transpose(1, 2) # [B, N, T, H]
        matrix_ac = torch.matmul(query_with_bias_u, key.transpose(-2, -1)) # [B, N, T, T] = [B, N, T, H] * [B, N, H, T]

        matrix_bd = self._RelPosBias(query + self.posv if not self.skip_term_b else self.posv, pos) # [B, N, T, T]

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        context, attn_weights = self.compute_context(value, scores, mask.unsqueeze(1) if mask is not None else None)

        return context, attn_weights

    def inference(self, inputs, mask, pos, cache=None):
        context, attn_weights = self.forward(inputs, mask, pos)
        return context, attn_weights, cache

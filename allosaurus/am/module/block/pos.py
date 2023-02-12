# File   : pos.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com


import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, emb_dim, scale_learnable=False, dropout=0.0):
        """Initialize class.

        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length

        """
        super(PositionalEncoding, self).__init__()
        self.emb_dim = emb_dim
        self.xscale = math.sqrt(self.emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.scale_learnable = scale_learnable

        if self.scale_learnable:
            self.alpha = nn.Parameter(torch.tensor(1.0))

    def _embedding_from_positions(self, position):
        """get absolute pos embedding based position.
        Args:
            position (torch.Tensor): Input. Its shape is (b, t)
        Returns:
            posemb (torch.Tensor): Encoded tensor. Its shape is (b, time, emb_dim)
        """
        batch_size, time_step = position.size()
        posemb = torch.zeros(batch_size, time_step, self.emb_dim, device=position.device)
        div_term = torch.exp(torch.arange(0, self.emb_dim, 2, device=position.device, dtype=torch.float32) * -(math.log(10000.0) / self.emb_dim))
        posemb[:, :, 0::2] = torch.sin(position.float().unsqueeze(-1) * div_term)
        posemb[:, :, 1::2] = torch.cos(position.float().unsqueeze(-1) * div_term)
        return posemb

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        pos = torch.arange(0, x.size(1), device=x.device).reshape(1, -1) # [1, t]
        posemb = self._embedding_from_positions(pos)  # [1, t, emb_dim]
        if self.scale_learnable:
            x = x + self.alpha * posemb
        else:
            x = x * self.xscale + posemb
        return self.dropout(x), posemb

    def forward_from_pos(self, x, pos):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (b, t, emb)
            pos (torch.Tensor), Its shape is (b, t)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (b, t, ...)
        """
        posemb = self._embedding_from_positions(pos)  # [b, t, emb_dim]
        if self.scale_learnable:
            x = x + self.alpha * posemb
        else:
            x = x * self.xscale + posemb
        return x, posemb

    def inference(self, x, step):
        raise NotImplementedError
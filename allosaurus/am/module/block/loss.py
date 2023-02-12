# File   : loss.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com


import torch
import torch.nn as nn
import torch.nn.functional as F
from otrans.data import PAD


class LabelSmoothingLoss(nn.Module):
    def __init__(self, size, smoothing=0.1, padding_idx=PAD, normalize_length=True):
        super().__init__()

        self.padding_idx = padding_idx
        self.smoothing = smoothing
        self.size = size
        self.normalize_length = normalize_length

    def forward(self, logits, target, mask=None):
        """LabelSmoothing Function with Mask

        Args:
            logits ([tensor]): logits with shape [batch, length, vocab_size]
            target ([tensor]): target with shape [batch, length]
            mask ([tensor], optional): mask tensor (bool) with shape [batch, length]
        """
        assert logits.dim() == 3 and logits.size(-1) == self.size
        
        pad_mask = target == self.padding_idx
        if mask is not None:
            mask = (pad_mask.int() + mask.int()) > 0
        else:
            mask = pad_mask

        logits = logits.reshape(-1, self.size)
        with torch.no_grad():
            confidence = logits.clone()
            confidence.fill_(self.smoothing / (self.size - 1))
            confidence.scatter_(1, target.reshape(-1).unsqueeze(1), 1 - self.smoothing)

        loss = torch.sum(F.kl_div(F.log_softmax(logits, dim=-1), confidence, reduction='none'), dim=-1)
        total = torch.sum(~mask)
        denom = total if self.normalize_length else logits.size(0)
        loss = torch.sum(loss.masked_fill_(mask.reshape(-1), 0.0)) / denom

        return loss



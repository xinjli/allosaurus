# File   : ffn.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


_ACTIVATION = {
    'relu': F.relu,
    'gelu': F.gelu,
    'glu': F.glu,
    'tanh': lambda x: torch.tanh(x),
    'swish': lambda x: x * torch.sigmoid(x)
}


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward
    """

    def __init__(self, d_model, d_ff, dropout, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = activation

        assert activation in ['relu', 'gelu', 'glu', 'tanh', 'swish']

        self.w_1 = nn.Linear(d_model, d_ff * 2 if activation == 'glu' else d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = _ACTIVATION[self.activation](x)
        return self.w_2(self.dropout(x))



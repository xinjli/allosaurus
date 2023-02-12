import torch
import torch.nn as nn

class AllosaurusCriterion(nn.Module):
    def __init__(self, config):
        super(AllosaurusCriterion, self).__init__()
        self.config = config
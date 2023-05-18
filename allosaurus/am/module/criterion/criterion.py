import torch
import torch.nn as nn

class AllospeechCriterion(nn.Module):
    def __init__(self, config):
        super(AllospeechCriterion, self).__init__()
        self.config = config
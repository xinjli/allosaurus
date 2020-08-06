import torch
import torch.nn as nn


def read_criterion(train_config):

    assert train_config.criterion == 'ctc', 'only ctc criterion is supported now'
    return CTCCriterion(train_config)


class CTCCriterion(nn.Module):
    def __init__(self, train_config):
        super().__init__()
        self.train_config = train_config

        self.logsoftmax = nn.LogSoftmax(dim=2)

        self.criterion = nn.CTCLoss(reduction='sum', zero_infinity=True)

    def forward(self,
                output_tensor: torch.tensor,
                output_lengths: torch.tensor,
                target_tensor: torch.tensor,
                target_lengths: torch.tensor):

        output_tensor = self.logsoftmax(output_tensor).transpose(0,1)
        loss = self.criterion(output_tensor, target_tensor, output_lengths, target_lengths)
        return loss
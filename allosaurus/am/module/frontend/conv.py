# File   : cnn.py
# Author : Zhengkun Tian
# Email  : zhengkun.tian@outlook.com
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def cal_width_dim_2d(input_dim, kernel_size, stride, padding=1):
    return math.floor((input_dim + 2 * padding - kernel_size)/stride + 1)


class Conv2dLayer(nn.Module):
    def __init__(self, input_size, in_channel, out_channel, kernel_size, stride,
                 dropout=0.1, batch_norm=False, residual=False, act_func_type='relu'):
        super(Conv2dLayer, self).__init__()

        self.input_size = input_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.batch_norm = batch_norm
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (0, kernel_size // 2 if isinstance(self.kernel_size, int) else kernel_size[1] // 2)

        self.residual = residual

        self.act_func_type = act_func_type

        self.conv_layer = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding)

        self.output_size = cal_width_dim_2d(input_size,
            self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[1],
            self.stride if isinstance(self.stride, int) else self.stride[1],
            padding=self.padding if isinstance(self.padding, int) else self.padding[1])

        if self.batch_norm:
            self.norm = nn.BatchNorm2d(out_channel)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """Forward computation.

        Args:
            x (FloatTensor): `[B, C_i, T, F]`
            mask (IntTensor): `[B, 1, T]`
        Returns:
            out (FloatTensor): `[B, C_o, T', F']`
            out_mask (IntTensor): `[B, 1, T]`

        """
        residual = x

        out = self.conv_layer(x)
        out = F.relu(out)

        if self.batch_norm:
            out = self.norm(out)

        out = self.dropout(out)

        if self.residual and out.size() == residual.size():
            out += residual

        mask = self.return_output_mask(mask, out.size(2))

        return out, mask

    def return_output_mask(self, mask, t):
        # conv1
        stride = self.stride if isinstance(self.stride, int) else self.stride[0]
        kernel_size = self.kernel_size if isinstance(self.kernel_size, int) else self.kernel_size[0]
        mask = mask[:, math.floor(kernel_size / 2)::stride][:,:t]
        return mask


class ConvFrontEnd(nn.Module):

    def __init__(self,
                 config,
                 input_size, output_size, in_channel=1, mid_channel=32,
                 out_channel=128, kernel_size=[[3,3],[3,3]], stride=[2, 2],
                 dropout=0.0, act_func_type='relu', front_end_layer_norm=False):
        super(ConvFrontEnd, self).__init__()
        self.config = config

        self.kernel_size = kernel_size
        self.stride = stride
        self.output_size = output_size

        self.act_func_type = act_func_type
        self.front_end_layer_norm = front_end_layer_norm

        assert isinstance(self.kernel_size, list) and len(self.kernel_size) == 2
        assert isinstance(self.stride, list) and len(self.stride) == 2

        self.conv1 = Conv2dLayer(
            input_size=input_size,
            in_channel=in_channel,
            out_channel=mid_channel,
            kernel_size=self.kernel_size[0],
            stride=self.stride[0],
            dropout=dropout,
            batch_norm=False,
            residual=False,
            act_func_type=act_func_type)

        self.conv2 = Conv2dLayer(
            self.conv1.output_size,
            in_channel=mid_channel,
            out_channel=out_channel,
            kernel_size=self.kernel_size[1],
            stride=self.stride[1],
            dropout=dropout,
            batch_norm=False,
            residual=False,
            act_func_type=act_func_type
        )

        self.conv_output_size = self.conv2.output_size * self.conv2.out_channel
        self.output_layer = nn.Linear(self.conv_output_size, self.output_size)

        if self.front_end_layer_norm:
            self.layer_norm = nn.LayerNorm(self.output_size)

    def forward(self, input_tensor, mask):
        """Subsample inputs

        :param torch.Tensor inputs: x tensor [batch, time, size]
        :param torch.Tensor inputs_mask: mask [batch, time]
        :return:
        
        """
        
        x = input_tensor.unsqueeze(1)
        x, mask = self.conv1(x, mask)
        x, mask = self.conv2(x, mask)
        
        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, c * f)
        x = self.output_layer(x)
    
        # x.masked_fill(~mask.unsqueeze(2), 0.0)

        if self.front_end_layer_norm:
            x = self.layer_norm(x)

        return x, mask

    def inference(self, x, mask, cache):

        x, mask = self.forward(x, mask)

        return x, mask, cache
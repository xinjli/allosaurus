# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import contextlib
import copy
import importlib.util
import math
import os
import sys
from typing import Callable, List
import warnings

import torch
import torch.nn.functional as F
import numpy as np

from itertools import accumulate
#from pyspeech.ml.torch.modules.gelu import gelu, gelu_accurate


# -*- coding: utf-8 -*-

"""Network related utility tools."""

import numpy as np
import torch
from allosaurus.utils.reporter import reporter


def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)


def pad_list(xs, pad_value=0, max_len=-1):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    n_batch = len(xs)

    if max_len == -1:
        max_len = max(x.size(0) for x in xs)

    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, :xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, max_len=None):
    """Function to make mask tensor containing indices of padded part
    example: lengths: [4,2,1]

    return
    array([[False, False, False, False],
       [False, False,  True,  True],
       [False,  True,  True,  True]])

    """
    if isinstance(lengths, torch.Tensor):
        if max_len is None:
            max_len = torch.max(lengths).item()
        batch_len = lengths.shape[0]

        position_tile = torch.arange(max_len, device=lengths.device).repeat(batch_len, 1)
        length_tile = lengths.repeat(max_len, 1).T

        return length_tile <= position_tile

    else:

        lengths = np.array(lengths)
        max_len = np.max(lengths)
        batch_len = lengths.shape[0]

        # (B, T)
        # e.g:
        # array([[0, 1, 2, 3],
        #        [0, 1, 2, 3],
        #        [0, 1, 2, 3]])
        position_tile = np.tile(np.arange(max_len), (batch_len, 1))

        # (B,T)
        # array([[4, 4, 4, 4],
        #        [2, 2, 2, 2],
        #        [1, 1, 1, 1]])
        length_tile = np.tile(lengths, (max_len, 1)).T

        return length_tile <= position_tile


def make_non_pad_mask(lengths, max_len=None):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.

    Examples:
        With only lengths.

        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]

        With the reference tensor.

        >>> xs = torch.zeros((3, 2, 4))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1],
                 [1, 1, 1, 1]],
                [[1, 1, 1, 0],
                 [1, 1, 1, 0]],
                [[1, 1, 0, 0],
                 [1, 1, 0, 0]]], dtype=torch.uint8)
        >>> xs = torch.zeros((3, 2, 6))
        >>> make_non_pad_mask(lengths, xs)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

        With the reference tensor and dimension indicator.

        >>> xs = torch.zeros((3, 6, 6))
        >>> make_non_pad_mask(lengths, xs, 1)
        tensor([[[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]],
                [[1, 1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1, 1],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0]]], dtype=torch.uint8)
        >>> make_non_pad_mask(lengths, xs, 2)
        tensor([[[1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1, 0]],
                [[1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0],
                 [1, 1, 1, 0, 0, 0]],
                [[1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0],
                 [1, 1, 0, 0, 0, 0]]], dtype=torch.uint8)

    """
    return ~make_pad_mask(lengths, max_len)


def mask_by_length(xs, lengths, fill=0):
    """Mask tensor according to length.

    Args:
        xs (Tensor): Batch of input tensor (B, `*`).
        lengths (LongTensor or List): Batch of lengths (B,).
        fill (int or float): Value to fill masked part.

    Returns:
        Tensor: Batch of masked input tensor (B, `*`).

    Examples:
        >>> x = torch.arange(5).repeat(3, 1) + 1
        >>> x
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5]])
        >>> lengths = [5, 3, 2]
        >>> mask_by_length(x, lengths)
        tensor([[1, 2, 3, 4, 5],
                [1, 2, 3, 0, 0],
                [1, 2, 0, 0, 0]])

    """
    assert xs.size(0) == len(lengths)
    ret = xs.data.new(*xs.size()).fill_(fill)
    for i, l in enumerate(lengths):
        ret[i, :l] = xs[i, :l]
    return ret


def th_accuracy(pad_outputs, pad_targets, ignore_label):
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax, D).
        ignore_label (int): Ignore label id.

    Returns:
        float: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(
        pad_targets.size(0),
        pad_targets.size(1),
        pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return float(numerator) / float(denominator)




def apply_to_tensor(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)

def apply_to_ndarray(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if isinstance(x, np.ndarray):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        elif isinstance(x, tuple):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def tensor_to_cuda(sample, device_id=0):

    if device_id == -1:
        return sample

    def _move_to_cuda(tensor):
        return tensor.to(device_id)

    return apply_to_tensor(_move_to_cuda, sample)

def ndarray_to_tensor(sample):

    def _move_to_tensor(dnarray):
        return torch.from_numpy(dnarray)

    return apply_to_ndarray(_move_to_tensor, sample)

def move_to_tensor(sample, device_id=-1):
    """
    move numpy array to torch tensor

    :param sample:
    :param device_id: -1 means cpu, other means gpu device_id
    :return:
    """

    sample = ndarray_to_tensor(sample)

    # move to cuda if device_id provided
    if device_id >= 0:
        sample = tensor_to_cuda(sample, device_id)

    return sample

def move_to_ndarray(sample):

    if sample.is_cuda:
        sample = sample.cpu()

    return sample.data.numpy()


def debug_tensor(origin_tensor, torch_name='tensor', verbose=False):

    torch_tensor = origin_tensor.float()

    if torch_tensor.grad is not None:
        reporter.success(f"{torch_name:50}({origin_tensor.type():20}) {str(torch_tensor.device):6} size:{str(list(torch_tensor.shape)):15} mean: {torch_tensor.mean().item(): 8.4f} std: {torch_tensor.std().item(): 8.4f} max: {torch_tensor.max().item(): 8.4f} min: {torch_tensor.min().item(): 8.4f} | grad mean: {torch_tensor.grad.mean().item(): 8.4f} std: {torch_tensor.grad.std().item(): 8.4f} max: {torch_tensor.grad.max().item(): 8.4f} min: {torch_tensor.grad.min().item(): 8.4f}" )
    else:
        reporter.success(f"{torch_name:50}({origin_tensor.type():20}) {str(torch_tensor.device):6} size:{str(list(torch_tensor.shape)):15} mean: {torch_tensor.mean().item(): 8.4f} std: {torch_tensor.std().item(): 8.4f} max: {torch_tensor.max().item(): 8.4f} min: {torch_tensor.min().item(): 8.4f} ")

    if verbose:
        reporter.success(f"{torch_name} contents: "+str(origin_tensor))


def debug_model(model):

    for name, parameter in model.named_parameters():
        debug_tensor(parameter, name)
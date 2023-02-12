import torch
import argparse
import logging
from copy import deepcopy
from dataclasses import dataclass, is_dataclass

import torch


def merge_with_parent(dc: dataclass, cfg: dict):

    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]

    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)



def load_converted_model(ckpt: str):
    ckpt_state = torch.load(ckpt, map_location="cpu")

    for required_key in ["task_cfg", "model_cfg", "model_weight"]:
        if required_key not in ckpt_state:
            raise ValueError(
                f"{ckpt} is not a valid checkpoint since the required key: {required_key} is missing"
            )

    task_cfg = merge_with_parent(AudioPretrainingConfig, ckpt_state["task_cfg"])
    model_cfg = merge_with_parent(Wav2Vec2Config, ckpt_state["model_cfg"])
    model = Wav2Vec2Model(model_cfg)
    model.load_state_dict(ckpt_state["model_weight"])
    return model, task_cfg

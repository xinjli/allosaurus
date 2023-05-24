from allosaurus.config import allosaurus_config
from allosaurus.utils.config import read_config
from pathlib import Path
import json
import argparse
from allosaurus.pm.module.mfcc import MFCC
from allosaurus.pm.module.raw import Raw
import torch

def read_pm(config_or_name, overwrite_config=None):

    # already instantiated
    if isinstance(config_or_name, PreprocessModel):
        return config_or_name

    # load configure
    if isinstance(config_or_name, str):
        config_path = Path(allosaurus_config.data_path / 'config' / 'pm' / (config_or_name + '.yml'))
        config = read_config(config_path, overwrite_config)

    else:
        config = config_or_name

    if config.model == 'mfcc':
        model = MFCC(config)
    else:
        assert config.model == 'raw'
        model = Raw(config)

    return PreprocessModel(model, config)


class PreprocessModel:

    def __init__(self, model, config):

        self.config = config
        self.model = model

    def compute(self, audio):

        feat = self.model.compute(audio)

        return feat
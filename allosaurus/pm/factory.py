from allosaurus.pm.mfcc import MFCC
import json
from argparse import Namespace

def read_pm(model_path, inference_config):
    """
    read feature extraction model

    :param pm_config:
    :return:
    """

    pm_config = Namespace(**json.load(open(str(model_path / 'pm_config.json'))))

    assert pm_config.model == 'mfcc_hires', 'only mfcc_hires is supported for allosaurus now'
    assert pm_config.backend == 'numpy', 'only numpy backend is supported for allosaurus now'

    model = MFCC(pm_config)
    return model
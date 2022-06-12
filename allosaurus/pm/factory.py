import json
from argparse import Namespace
import yaml
from allosaurus.utils.config import dotdict

def read_pm(model_path, inference_config):
    """
    read feature extraction model

    :param pm_config:
    :return:
    """

    if (model_path / 'pm_config.json').exists():
        pm_config = Namespace(**json.load(open(str(model_path / 'pm_config.json'))))
    elif (model_path / 'pm_config.yml').exists():
        pm_config = dotdict(yaml.load(open(str(model_path / 'pm_config.yml')), Loader=yaml.FullLoader))

    if pm_config.backend == 'numpy':
        assert pm_config.model == 'mfcc_hires', 'only mfcc_hires is supported for allosaurus now'

        from allosaurus.pm.numpy.mfcc import NumpyMFCC
        model = NumpyMFCC(pm_config)

    else:
        assert pm_config.backend == 'torch', 'only numpy backend is supported for allosaurus now'

        from allosaurus.pm.torch.mfcc import TorchMFCC
        model = TorchMFCC(pm_config)

    return model
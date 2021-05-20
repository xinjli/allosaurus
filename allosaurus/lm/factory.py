from allosaurus.lm.decoder import PhoneDecoder
import json
from argparse import Namespace

def read_lm(model_path, inference_config):
    """
    read language model (phone inventory)

    :param pm_config:
    :return:
    """

    lm_config = Namespace(**json.load(open(str(model_path / 'lm_config.json'))))
    assert lm_config.model   == 'phone_ipa', 'only phone_ipa model is supported for allosaurus now'
    assert lm_config.backend == 'numpy', 'only numpy backend is supported for allosaurus now'

    pm_config = Namespace(**json.load(open(str(model_path / 'pm_config.json'))))

    # import configs from pm config
    inference_config.window_size = pm_config.window_size + (pm_config.feature_window - 1)*pm_config.window_shift
    inference_config.window_shift = pm_config.window_shift*pm_config.feature_window

    model = PhoneDecoder(model_path, inference_config)
    return model
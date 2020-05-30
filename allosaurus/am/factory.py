from allosaurus.am.allosaurus_torch import AllosaurusTorchModel
from allosaurus.am.utils import *
import json
from argparse import Namespace

def read_am(model_path, inference_config):
    """
    load pretrained acoustic model

    :param model_path: path to the
    :return:
    """

    am_config = Namespace(**json.load(open(str(model_path / 'am_config.json'))))

    assert am_config.model == 'allosaurus', "This project only support allosaurus model"

    model = AllosaurusTorchModel(am_config)

    # load weights
    torch_load(model, str(model_path / 'model.pt'), inference_config.device_id)

    return model
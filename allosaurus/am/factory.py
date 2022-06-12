from allosaurus.am.model.allosaurus import AllosaurusModel
from allosaurus.am.model.compositional_phonetics import CompositionalPhoneticsModel
from allosaurus.am.utils import *
from allosaurus.lm.inventory import Inventory
from allosaurus.lm.unit import write_unit
from allosaurus.utils.config import dotdict
import json
from argparse import Namespace
from allosaurus.model import get_model_path
import yaml

def read_am(model_path, inference_config):
    """
    load pretrained acoustic model

    :param model_path: path to the
    :return:
    """

    if (model_path / 'am_config.json').exists():
        am_config = Namespace(**json.load(open(str(model_path / 'am_config.json'))))
    elif (model_path / 'am_config.yml').exists():
        am_config = dotdict(yaml.load(open(str(model_path / 'am_config.yml')), Loader=yaml.FullLoader))
        am_config.model_path = model_path

    assert am_config.model in ['allosaurus', 'compositional'], "This project only support the original allosaurus model and compositional phonetics model"

    if am_config.model == 'allosaurus':
        model = AllosaurusModel(am_config)
    else:
        assert am_config.model == 'compositional'
        model = CompositionalPhoneticsModel(am_config)

    # load weights
    torch_load(model, str(model_path / 'model.pt'), inference_config.device_id)

    return model

def transfer_am(train_config):
    """
    initialize the acoustic model with a pretrained model for fine-tuning

    :param model_path: path to the
    :return:
    """

    pretrained_model_path = get_model_path(train_config.pretrained_model)

    am_config = Namespace(**json.load(open(str(pretrained_model_path / 'am_config.json'))))

    assert am_config.model == 'allosaurus', "Fine-tuning feature only support allosaurus model for now"

    # load inventory
    inventory = Inventory(pretrained_model_path)

    # get unit_mask which maps the full phone inventory to the target phone inventory
    unit_mask = inventory.get_mask(train_config.lang, approximation=True)

    # reset the new phone_size
    am_config.phone_size = len(unit_mask.target_unit)

    model = AllosaurusTorchModel(am_config)

    # load the pretrained model and setup the phone_layer with correct weights
    torch_load(model, str(pretrained_model_path / 'model.pt'), train_config.device_id, unit_mask)

    # update new model
    new_model = train_config.new_model

    # get its path
    model_path = get_model_path(new_model)

    # overwrite old am_config
    new_am_config_json = vars(am_config)
    json.dump(new_am_config_json, open(str(model_path / 'am_config.json'), 'w'), indent=4)

    # overwrite old phones
    write_unit(unit_mask.target_unit, model_path / 'phone.txt')

    # overwrite old model
    torch_save(model, model_path / 'model.pt')

    return model
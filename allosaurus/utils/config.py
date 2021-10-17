from pathlib import Path
import json
from argparse import Namespace
from allospeech.config import allospeech_config
import yaml

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def read_config(config_name_or_path, overwrite_config=None):
    assert isinstance(config_name_or_path, str) or isinstance(config_name_or_path, Path), config_name_or_path+' is not valid'

    if isinstance(config_name_or_path, Path) or config_name_or_path.endswith('config.yml'):
        # load utterances from a yaml file
        config_dict = yaml.load(open(str(config_name_or_path)), Loader=yaml.FullLoader)

    else:
        # load preset config from config trunk
        config_yml = config_name_or_path + '.yml'
        config_path = allospeech_config.data_path / 'config' / config_yml

        assert config_path.exists()
        config_dict = yaml.load(open(str(config_path)), Loader=yaml.FullLoader)

    if overwrite_config is not None:
        for k,v in overwrite_config.items():
            if k not in config_dict:
                print('WARNING: ', k, ' not found !')
                config_dict[k] = v
            else:
                print("overwriting ", config_dict[k], ' --> ', v)
                config_dict[k] = v

    args = dotdict(config_dict)

    return args

def write_config(config, config_path):

    write_dict = {}
    for k,v in config.items():

        if k in ['am', 'lm', 'pm']:
            write_dict[k] = dict(v.config)
        elif isinstance(v, Path):
            write_dict[k] = str(v)
        else:
            write_dict[k] = v

    w = open(config_path, 'w')
    data = yaml.dump(write_dict, Dumper=yaml.Dumper)
    w.write(data)
    w.close()

def merge_config(config_a, config_b):
    dict_a = vars(config_a)
    dict_b = vars(config_b)
    dict_a.update(dict_b)

    return Namespace(**dict_a)

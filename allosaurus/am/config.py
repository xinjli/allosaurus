import sys
from allosaurus.utils.config import *
from allosaurus.config import allosaurus_config
import yaml
import time

def read_exp_config(exp_name):
    args = dotdict({'rank': 0, 'model_path': 'auto', 'force_deploy': False})
    exp_path = allosaurus_config.data_path / 'exp' / (str(exp_name) + '.yml')

    if not exp_path.exists():
        exp_path = allosaurus_config.data_path / 'exp' / 'research' / (str(exp_name) + '.yml')

    config_dict = yaml.load(open(str(exp_path)), Loader=yaml.FullLoader)
    args.update(config_dict)
    return args


def parse_args():

    args = dotdict({'rank': 0, 'model_path': 'auto', 'force_deploy': False})

    for item in sys.argv:
        if item.startswith('--'):
            key, val = item[2:].split('=')

            # if val in ['True', 'true']:
            #     val = True
            # elif val in ['False', 'false']:
            #     val = False
            # elif '/' in val:
            #     val = str(val)
            # elif '.' in val:
            #     val = float(val)
            # elif val.isdigit():
            #     val = int(val)

            # handle special args
            if key == 'corpus_ids':
                val = val.split(',')

            if key == 'exp':
                config_dict = read_exp_config(val)
                args.update(config_dict)
                args['exp'] = val

            if key == 'ngpu':
                val = int(val)

            args[key] = val

    args.model_path = get_model_path(args)

    return dotdict(args)


def config_to_dict(config):
    config_dict = vars(config)
    config_dict["model_path"] = str(config_dict["model_path"])

    return config_dict


def get_model_path(args):

    if args.model_path ==  'none':
        return None
    elif args.model_path == 'auto':

        model_name = args.exp

        if model_name.startswith('research/'):
            model_name = model_name[9:]

        return allosaurus_config.model_path / model_name
    else:
        return Path(args.model_path)


def create_am_config(input_args=None):

    args = parse_args()

    print(args)
    # load predefined loader config
    #task_config = read_config('am/task/'+args.task)
    #model_config = read_config('am/arch/'+args.arch)

    #task_config.update(model_config)
    #task_config.update(args)
    #task_config.arch = model_config.arch

    #args = task_config

    return args


def read_am_config(config_name_or_path):

    assert isinstance(config_name_or_path, str) or isinstance(config_name_or_path, Path)

    if isinstance(config_name_or_path, Path) or config_name_or_path.endswith('config.json'):
        # load utterances from a json file
        config_dict = json.load(open(str(config_name_or_path)))

    else:
        # load preset config from config trunk
        config_json = config_name_or_path + '.json'
        config_path = allosaurus_config.data_path / 'config/am' / config_json

        assert config_path.exists()

        config_dict = json.load(open(str(config_path)))


    args = dotdict(config_dict)
    print("Loaded ", args)

    return args


def write_am_config(config, config_path):
    # serialize utterances to a json

    config["model_path"] = str(config["model_path"])
    json.dump(config, open(str(config_path), 'w', encoding='utf-8'), indent=4)
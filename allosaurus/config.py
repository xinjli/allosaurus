from pathlib import Path
import time
import torch.distributed
import platform
import yaml

class AllosaurusConfigure:

    root_path = Path(__file__).parent

    # data path
    data_path = root_path / 'data'
    data_str = str(data_path)

    node_name = platform.node()

    model_path = data_path / 'arch'
    config_path = data_path / 'config'
    corpus_path = data_path

    def __init__(self):

        setting_dicts = yaml.load(open(self.config_path / 'path.yml', 'rb'), Loader=yaml.FullLoader)

        nodename = platform.node()

        self.database_path = Path('/tmp')
        self.corpora_path = Path('/tmp')
        self.tmp_path = Path('/tmp')

        for hostname, setting_dict in setting_dicts.items():
            if nodename.endswith(hostname):
                self.database_path = Path(setting_dict['database_path'])
                self.corpora_path = self.database_path / 'corpus'
                self.tmp_path = Path(setting_dict['tmp_path'])

        self.model_path = self.root_path / 'data/model'


    def get_world_size(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_world_size()
        else:
            return 1

    def get_rank(self):
        if torch.distributed.is_initialized():
            return torch.distributed.get_rank()
        else:
            return 0

    def get_timestamp(self):
        return time.strftime("%y%m%d_%H%M")


allosaurus_config = AllosaurusConfigure()
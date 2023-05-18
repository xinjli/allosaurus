from allosaurus.config import allosaurus_config
from allosaurus.utils.config import read_config
from allosaurus.lm.cleaner import Cleaner
from allosaurus.lm.module.phone import PhoneTokenizer
from pathlib import Path
import json
import yaml


def read_lm(config_or_name):
    """
    read language arch (cleaner, tokenizer)

    :param config:
    :return:
    """

    # already get instantiated
    if isinstance(config_or_name, LanguageModel):
        return config_or_name

    if isinstance(config_or_name, str):
        config_path = Path(allosaurus_config.data_path / 'config' / 'lm' / (config_or_name + '.yml'))
        config = read_config(config_path)
    else:
        config = config_or_name

    return LanguageModel(config)


class LanguageModel:

    def __init__(self, config):

        self.model = config.model
        self.config = config
        self.default_lang_id = self.config.default_lang_id

        self.cleaner = Cleaner(config)
        self.tokenizer = PhoneTokenizer(config)

    def tokenize(self, sent, lang_id=None):
        
        if lang_id is None:
            lang_id = self.default_lang_id

        cleaned_sent = self.cleaner.compute(sent, lang_id)
        ipa_lst = self.tokenizer.tokenize(cleaned_sent, lang_id)

        return ipa_lst

    def compute(self, sent, lang_id=None):

        if lang_id is None:
            lang_id = self.default_lang_id

        cleaned_sent = self.cleaner.compute(sent, lang_id)
        id_lst = self.tokenizer.compute(cleaned_sent, lang_id)

        return id_lst

    def decode(self, id_lst, lang_id=None):
        return self.tokenizer.decode(id_lst, lang_id)
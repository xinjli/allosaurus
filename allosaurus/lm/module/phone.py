from transphone.tokenizer import read_tokenizer
from phonepiece.inventory import read_inventory
import torch

class PhoneTokenizer:

    def __init__(self, config):

        self.config = config

        if config.model == 'base_phone':
            self.base = True
        else:
            self.base = False

        self.lang2tokenizer = {}
        self.lang2inventory = {}

    def __repr__(self):
        return f"<PhoneTokenizer />"

    def tokenize(self, sent, lang_id='eng'):

        if lang_id not in self.lang2tokenizer:
            tokenizer = read_tokenizer(lang_id, device='cpu')
            self.lang2tokenizer[lang_id] = tokenizer

        tokenizer = self.lang2tokenizer[lang_id]

        ipa_lst = tokenizer.tokenize(sent)
        return ipa_lst

    def compute(self, sent, lang_id='eng'):

        ipa_lst = self.tokenize(sent, lang_id)

        if lang_id not in self.lang2inventory:
            inventory = read_inventory(lang_id, base=self.base)
            self.lang2inventory[lang_id] = inventory

        inventory = self.lang2inventory[lang_id]

        id_lst = inventory.phoneme.atoi(ipa_lst)
        return torch.LongTensor(id_lst)

    def decode(self, id_lst, lang_id='eng'):

        if lang_id not in self.lang2inventory:
            inventory = read_inventory(lang_id)
            self.lang2inventory[lang_id] = inventory

        inventory = self.lang2inventory[lang_id].phoneme
        return inventory.itoa(id_lst)
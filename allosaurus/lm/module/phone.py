from transphone.tokenizer import read_tokenizer
from phonepiece.inventory import read_inventory
import torch
import itertools

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

    def decode(self, id_lst_or_info_list, lang_id='eng'):

        if len(id_lst_or_info_list) == 0:
            return []

        if lang_id not in self.lang2inventory:
            inventory = read_inventory(lang_id)
            self.lang2inventory[lang_id] = inventory

        inventory = self.lang2inventory[lang_id].phoneme

        if isinstance(id_lst_or_info_list[0], int):

            id_lst = id_lst_or_info_list

            if lang_id != 'jpn':
                id_lst = [x[0] for x in itertools.groupby(id_lst)]

            phoneme_lst = inventory.itoa(id_lst)
            return phoneme_lst

        else:
            assert isinstance(id_lst_or_info_list[0], dict)

            id_lst = [x['phone_id'] for x in id_lst_or_info_list]
            phoneme_lst = inventory.itoa(id_lst)

            for i, info in enumerate(id_lst_or_info_list):
                info['phone'] = phoneme_lst[i]

                if 'alternative_ids' in info:
                    info['alternative_phones'] = inventory.itoa(info['alternative_ids'])

            return id_lst_or_info_list
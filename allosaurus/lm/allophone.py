from phonepiece.config import *
from allosaurus.lm.unit import read_unit
import json
from collections import defaultdict


def read_allophone(model_path, lang_id):

    if (Path(lang_id) / "allophone.txt").exists():
        lang_dir = Path(lang_id)
    else:
        lang_dir = model_path / 'inventory' / lang_id

    phone_unit = read_unit(lang_dir / 'phone.txt')
    phoneme_unit = read_unit(lang_dir / 'phoneme.txt')
    phone2phoneme = defaultdict(list)

    # use allophone file if allovera supports it

    for line in open(lang_dir / 'allophone.txt', encoding='utf-8'):
        fields = line.strip().split()
        phoneme = fields[0]

        for phone in fields[1:]:
            phone2phoneme[phone].append(phoneme)

    # some placeholder
    phone2phoneme['<blk>'] = ['<blk>']
    phone2phoneme['<eos>'] = ['<eos>']

    return Allophone(lang_id, phoneme_unit, phone_unit, phone2phoneme)


class Allophone:

    def __init__(self, lang_id, phoneme, phone, phone2phoneme):
        self.lang_id = lang_id
        self.phoneme = phoneme
        self.phone = phone
        self.phone2phoneme = phone2phoneme

        self.articulatory = None
        self.nearest_mapping = dict()

    def __str__(self):
        return f"<Allophone {self.lang_id} phoneme: {len(self.phoneme)}, phone: {len(self.phone)}>"

    def __repr__(self):
        return self.__str__()
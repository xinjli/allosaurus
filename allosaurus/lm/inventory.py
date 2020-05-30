import json
from allosaurus.lm.mask import *

class Inventory:

    def __init__(self, model_path):

        self.model_path = model_path

        self.lang_names = []

        self.lang2phonefile = dict()

        # load all available inventories
        langs = json.load(open(self.model_path / 'inventory' / 'index.json', 'r', encoding='utf-8'))

        # load all phones list
        self.unit = read_unit(str(self.model_path / 'phone.txt'))

        for lang in langs:

            lang_name = lang['LanguageName']
            phone_text = lang['phonelists']
            self.lang_names.append(lang_name.lower())

            self.lang2phonefile[lang_name.lower()] = phone_text


    def get_mask(self, lang_id=None, approximation=False):

        assert lang_id in self.lang2phonefile, "Language "+lang_id+" is not available !"

        unit_file = self.model_path / 'inventory' / self.lang2phonefile[lang_id]

        target_unit = read_unit(str(unit_file))

        return UnitMask(self.unit, target_unit, approximation)
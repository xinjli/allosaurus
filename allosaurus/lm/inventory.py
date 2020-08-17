import json
from allosaurus.lm.mask import *


class Inventory:

    def __init__(self, model_path):

        self.model_path = model_path

        self.lang_names = []
        self.lang_ids = []
        self.glotto_ids = []

        self.lang2phonefile = dict()

        # load all available inventories
        langs = json.load(open(self.model_path / 'inventory' / 'index.json', 'r', encoding='utf-8'))

        # load all phones list
        self.unit = read_unit(str(self.model_path / 'phone.txt'))

        for lang in langs:

            lang_name = lang['LanguageName']
            iso_id = lang['ISO6393']
            glotto_id = lang['GlottoCode']
            phone_text = lang['phonelists']

            self.lang_names.append(lang_name.lower())
            self.lang_ids.append(iso_id.lower())
            self.glotto_ids.append(glotto_id)

            # register both iso_id and glottocode id
            self.lang2phonefile[iso_id.lower()] = phone_text
            self.lang2phonefile[glotto_id.lower()] = phone_text

    def get_unit(self, lang_id):
        """
        load a unit specified by the lang_id

        Args:
            lang_id: ISO id

        Returns:

        """

        assert lang_id in self.lang2phonefile, "Language "+lang_id+" is not available !"

        # search customized file first, if not exist use the default one.
        updated_unit_file = self.model_path / 'inventory' / ('updated_'+self.lang2phonefile[lang_id])
        if updated_unit_file.exists():
            unit_file = updated_unit_file
        else:
            unit_file = self.model_path / 'inventory' / self.lang2phonefile[lang_id]

        target_unit = read_unit(str(unit_file))

        return target_unit

    def update_unit(self, lang_id, unit_file):
        """
        update the existing unit with a new unit file

        Args:
            lang_id:
            unit_file:

        Returns:

        """

        assert lang_id in self.lang2phonefile, "Language "+lang_id+" is not available !"


        # load the new unit file and validate its format
        new_unit = read_unit(unit_file)

        # the model path it should be stored
        updated_unit_file = self.model_path / 'inventory' / ('updated_'+self.lang2phonefile[lang_id])

        # save the new file
        write_unit(new_unit, updated_unit_file)

    def restore_unit(self, lang_id):
        """
        restore the original phone units

        Args:
            lang_id:

        Returns:

        """

        assert lang_id in self.lang2phonefile, "Language "+lang_id+" is not available !"

        # the updated unit file
        updated_unit_file = self.model_path / 'inventory' / ('updated_'+self.lang2phonefile[lang_id])

        # check whether it has an updated file
        assert updated_unit_file.exists(), "language "+lang_id+" does not have any customized inventory."

        # delete this file
        updated_unit_file.unlink()

    def get_mask(self, lang_id=None, approximation=False):

        target_unit = self.get_unit(lang_id)

        return UnitMask(self.unit, target_unit, approximation)
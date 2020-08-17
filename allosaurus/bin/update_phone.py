from pathlib import Path
from allosaurus.lm.inventory import Inventory
from allosaurus.model import get_model_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Update language inventory')
    parser.add_argument('-l', '--lang',  type=str, required=True, help='specify which language inventory to update.')
    parser.add_argument('-m', '--model', type=str, default='latest', help='specify which model inventory')
    parser.add_argument('-i', '--input', type=str, required=True, help='your new inventory file')

    args = parser.parse_args()

    model_path = get_model_path(args.model)

    inventory = Inventory(model_path)

    lang = args.lang

    # verify lang is not ipa as it is an alias to the entire inventory
    assert args.lang != 'ipa', "ipa is not a proper lang to update. use list_lang to find a proper language"

    assert lang.lower() in inventory.lang_ids or lang.lower() in inventory.glotto_ids, f'language {args.lang} is not supported. Please verify it is in the language list'

    new_unit_file = Path(args.input)

    # check existence of the file
    assert new_unit_file.exists(), args.input+' does not exist'

    # update this new unit
    inventory.update_unit(lang, new_unit_file)
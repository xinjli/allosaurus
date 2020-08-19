from pathlib import Path
from allosaurus.model import get_model_path
from allosaurus.lm.inventory import Inventory
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('List language phone inventory')
    parser.add_argument('-l', '--lang',  type=str, default='ipa',         help='specify which language inventory to use for recognition. default "ipa" is to use all phone inventory')
    parser.add_argument('-m', '--model', type=str, default='latest',     help='specify which model inventory')

    args = parser.parse_args()
    model_path = get_model_path(args.model)

    inventory = Inventory(model_path)

    print("Available Languages")
    for lang_id, glotto_id, lang_name in zip(inventory.lang_ids, inventory.glotto_ids, inventory.lang_names):
        lang_name = lang_name.encode('ascii', 'ignore')
        print('- ISO639-3: ', lang_id, 'Glotto Code', glotto_id, ' name: ', lang_name)

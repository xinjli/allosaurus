from pathlib import Path
from allosaurus.lm.inventory import Inventory
from allosaurus.model import get_model_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('List language phone inventory')
    parser.add_argument('-l', '--lang', type=str,  default='ipa',         help='specify which language inventory to use for recognition. default "ipa" is to use all phone inventory')
    parser.add_argument('-m', '--model', type=str, default='latest',     help='specify which model inventory')
    parser.add_argument('-a', '--approximate', type=bool, default=False, help='the phone inventory can still hardly to cover all phones. You can use turn on this flag to map missing phones to other similar phones to recognize. The similarity is measured with phonological features')

    args = parser.parse_args()

    model_path = get_model_path(args.model)

    inventory = Inventory(model_path)

    if args.lang == 'ipa':
        print("available phones: ", list(inventory.unit.id_to_unit.values())[1:])
    else:
        lang = args.lang
        assert lang.lower() in inventory.lang_ids or lang.lower() in inventory.glotto_ids, f'language {args.lang} is not supported. Please verify it is in the language list'

        mask = inventory.get_mask(args.lang.lower(), approximation=args.approximate)

        unit = mask.target_unit
        print('available phones: ', list(unit.id_to_unit.values())[1:])

        if args.approximate:
            mask.print_maps()
from pathlib import Path
from allosaurus.lm.inventory import Inventory
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('List language phone inventory')
    parser.add_argument('-l', '--lang', type=str, default='ipa', help='specify which language inventory to use for recognition. default "ipa" is to use all phone inventory')

    args = parser.parse_args()

    model_dir = Path(__file__).parent / 'pretrained'

    models = sorted(model_dir.glob('*'))
    if len(models) == 0:
        print("No models are available, you can maually download a model with download command or just run inference to download the latest one automatically")
        exit(0)

    inventory = Inventory(models[0])

    if args.lang == 'ipa':
        print(list(inventory.unit.id_to_unit.values())[1:])
    else:
        assert args.lang.lower() in inventory.lang_ids, f'language {args.lang} is not supported. Please verify it is in the language list'

        unit = inventory.get_mask(args.lang.lower()).target_unit
        print(list(unit.id_to_unit.values())[1:])
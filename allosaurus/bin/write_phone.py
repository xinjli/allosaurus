from pathlib import Path
from allosaurus.lm.inventory import Inventory
from allosaurus.lm.unit import write_unit
from allosaurus.model import get_model_path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Write out current phone file')
    parser.add_argument('-l', '--lang',   type=str, required=True, help='specify which language inventory to update.')
    parser.add_argument('-m', '--model',  type=str, default='latest', help='specify which model inventory')
    parser.add_argument('-o', '--output', type=str, required=True, help='write out your current phone file.')
    parser.add_argument('-f', '--format', type=str, default='simple', choices=['simple', 'kaldi'], help='select your output format')

    args = parser.parse_args()

    model_path = get_model_path(args.model)

    inventory = Inventory(model_path)

    lang = args.lang

    unit = inventory.get_unit(lang)
    write_unit(unit, args.output, args.format)
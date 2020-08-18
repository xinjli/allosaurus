import argparse
from pathlib import Path
from allosaurus.model import resolve_model_name
from allosaurus.lm.inventory import *
from tqdm import tqdm

def prepare_token(data_path, model, lang_id):

    model_path = Path(__file__).parent.parent / 'pretrained' / model

    #assert model_path.exists(), f"{model} is not a valid model"

    inventory = Inventory(model_path)
    unit = inventory.get_unit(lang_id)

    writer = open(str(data_path / 'token'), 'w', encoding='utf-8')

    for line in tqdm(open(data_path / 'text', 'r', encoding='utf-8').readlines()):
        fields = line.strip().split()
        utt_id = fields[0]

        phones = fields[1:]

        id_lst = unit.get_ids(phones)

        writer.write(utt_id+' '+' '.join(map(str, id_lst))+'\n')

    writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('allosaurus tool to assign token id for fine-tuning')
    parser.add_argument('--path', required=True, type=str, help='path to the directory containing the text file')
    parser.add_argument('--model', type=str, default='latest', help='specify the model you want to fine-tune')
    parser.add_argument('--lang', type=str, default='epi', help='specify the ISO language id for your target language')

    args = parser.parse_args()
    data_path = Path(args.path)

    text_path = data_path / 'text'

    assert text_path.exists(), "the path directory should contain a text file, please check README.md for details"

    # resolve model's name
    model_name = resolve_model_name(args.model)
    if model_name == "none":
        print("Model ", model_name, " does not exist. Please download this model or use an existing model in list_model")
        exit(0)

    args.model = model_name

    # extract token
    prepare_token(data_path, args.model, args.lang)
from pathlib import Path
from allosaurus.lm.inventory import Inventory

if __name__ == '__main__':

    model_dir = Path(__file__).parent / 'pretrained'

    models = sorted(model_dir.glob('*'))
    if len(models) == 0:
        print("No models are available, you can maually download a model with download command or just run inference to download the latest one automatically")
        exit(0)

    inventory = Inventory(models[0])

    print("Available Languages")
    for lang_id, lang_name in zip(inventory.lang_ids, inventory.lang_names):
        print('- language id: ',lang_id, ' name: ', lang_name)
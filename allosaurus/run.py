from allosaurus.app import read_recognizer
from allosaurus.model import get_all_models, resolve_model_name
from allosaurus.bin.download_model import download_model
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Allosaurus phone recognizer')
    parser.add_argument('-d', '--device_id', type=int, default=-1, help='specify cuda device id to use, -1 means no cuda and will use cpu for inference')
    parser.add_argument('-m', '--model', type=str, default='latest', help='specify which model to use. default is to use the latest local model')
    parser.add_argument('-l', '--lang', type=str,  default='ipa',help='specify which language inventory to use for recognition. default is to use all phone inventory')
    parser.add_argument('-i', '--input', type=str, required=True, help='specify your input wav file')
    parser.add_argument('-a', '--approximate', type=bool, default=False, help='the phone inventory can still hardly to cover all phones. You can use turn on this flag to map missing phones to other similar phones to recognize. The similarity is measured with phonological features')

    args = parser.parse_args()

    # check file format
    assert args.input.endswith('.wav'), " Error: Please use a wav file. other audio files can be converted to wav by sox"

    # download specified model automatically if no model exists
    if len(get_all_models()) == 0:
        download_model('latest')

    # resolve model's name
    model_name = resolve_model_name(args.model)
    if model_name == "none":
        print("Model ", model_name, " does not exist. Please download this model or use an existing model in list_model")
        exit(0)

    args.model = model_name

    # create recognizer
    recognizer = read_recognizer(args)

    # run inference
    phones = recognizer.recognize(args.input, args.lang)

    print(phones)
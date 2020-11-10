from allosaurus.app import read_recognizer
from allosaurus.model import get_all_models, resolve_model_name
from allosaurus.bin.download_model import download_model
from pathlib import Path
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser('Allosaurus phone recognizer')
    parser.add_argument('-d', '--device_id', type=int, default=-1, help='specify cuda device id to use, -1 means no cuda and will use cpu for inference')
    parser.add_argument('-m', '--model', type=str, default='latest', help='specify which model to use. default is to use the latest local model')
    parser.add_argument('-l', '--lang', type=str,  default='ipa',help='specify which language inventory to use for recognition. default is to use all phone inventory')
    parser.add_argument('-i', '--input', type=str, required=True, help='specify your input wav file/directory')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='specify output file. the default will be stdout')
    parser.add_argument('-k', '--topk', type=int, default=1, help='output k phone for each emitting frame')
    parser.add_argument('-a', '--approximate', type=bool, default=False, help='the phone inventory can still hardly to cover all phones. You can use turn on this flag to map missing phones to other similar phones to recognize. The similarity is measured with phonological features')

    args = parser.parse_args()

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

    # output file descriptor
    output_fd = None
    if args.output != 'stdout':
        output_fd = open(args.output, 'w', encoding='utf-8')

    # input file/path
    input_path = Path(args.input)

    if input_path.is_dir():
        wav_list = sorted(list(input_path.glob('*.wav')))
        for wav_path in wav_list:
            phones = recognizer.recognize(str(wav_path), args.lang)

            # save to file or print to stdout
            if output_fd:
                output_fd.write(wav_path.name+' '+phones+'\n')
            else:
                print(wav_path.name+' '+phones)

    else:

        # check file format
        assert args.input.endswith('.wav'), " Error: Please use a wav file. other audio files can be converted to wav by sox"

        # run inference
        phones = recognizer.recognize(args.input, args.lang, args.topk)

        if output_fd:
            output_fd.write(phones+'\n')
        else:
            print(phones)

    if output_fd:
        output_fd.close()
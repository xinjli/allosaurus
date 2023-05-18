import os
from allosaurus.app import read_recognizer
import argparse
from allosaurus.utils.checkpoint_utils import resolve_model_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser('a utility to run recognizer')
    parser.add_argument('-m', '--model', default='latest', help='specify which arch to download. A list of downloadable models are available on Github')
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('-l', '--lang', default='eng', help='language')
    parser.add_argument('-i', '--input', type=str, required=True, help='specify your input wav file/directory')
    parser.add_argument('-o', '--output', type=str, default='stdout', help='specify output file or directory. the default will be stdout')
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-d', '--segment_duration', type=int, default=15)

    args = parser.parse_args()

    batch_size = int(args.batch_size)
    segment_duration = args.segment_duration
    input_path = args.input
    output = args.output

    lang = args.lang

    model_name, checkpoint = resolve_model_name(args.model, args.checkpoint)
    model = read_recognizer(model_name, checkpoint)

    file_format = 'kaldi'
    if os.path.isfile(input_path):
        file_format = 'text'

    model.recognize_batch(input_path, lang, output, batch_size=batch_size, segment_duration=segment_duration, format=file_format)
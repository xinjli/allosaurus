import argparse
from pathlib import Path
from allosaurus.model import resolve_model_name
from allosaurus.audio import read_audio
from allosaurus.pm.factory import read_pm
from allosaurus.pm.kdict import KaldiWriter
from tqdm import tqdm

def prepare_feature(data_path, model):

    model_path = Path(__file__).parent.parent / 'pretrained' / model

    # create pm (pm stands for preprocess model: audio -> feature etc..)
    pm = read_pm(model_path, None)

    # data path should be pointing the absolute path
    data_path = data_path.absolute()

    # writer for feats
    feat_writer = KaldiWriter(data_path / 'feat')

    # writer for the shape of each utterance
    # format: utt_id shape[0] shape[1]
    shape_writer = open(data_path / 'shape', 'w')

    for line in tqdm(open(data_path / 'wave', 'r', encoding='utf-8').readlines()):
        fields = line.strip().split()
        utt_id = fields[0]
        audio_path = fields[1]

        assert Path(audio_path).exists(), audio_path+" does not exist!"

        audio = read_audio(audio_path)

        # extract feature
        feat = pm.compute(audio)

        # write shape
        shape_writer.write(f'{utt_id} {feat.shape[0]} {feat.shape[1]}\n')

        feat_writer.write(utt_id, feat)

    feat_writer.close()
    shape_writer.close()

if __name__ == '__main__':

    parser = argparse.ArgumentParser('allosaurus tool to extract audio feature for fine-tuning')
    parser.add_argument('--path', required=True, type=str, help='path to the directory containing the wave file')
    parser.add_argument('--model', type=str, default='latest', help='specify the model you want to fine-tune')

    args = parser.parse_args()
    data_path = Path(args.path)

    wave_path = data_path / 'wave'

    assert wave_path.exists(), "the path directory should contain a wave file, please check README.md for details"

    # resolve model's name
    model_name = resolve_model_name(args.model)
    if model_name == "none":
        print("Model ", model_name, " does not exist. Please download this model or use an existing model in list_model")
        exit(0)

    args.model = model_name

    # extract feature
    prepare_feature(data_path, args.model)




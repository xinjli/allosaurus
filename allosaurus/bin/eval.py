from pathlib import Path
from allosaurus.app import read_recognizer
import shutil
from tqdm import tqdm
import editdistance
import argparse
from allosaurus.utils.checkpoint_utils import find_topk_models
from allosaurus.corpus import read_corpus_path
from transphone.tokenizer import read_tokenizer
import os
import panphon.distance
from allosaurus.config import allosaurus_config


def get_exp_dir(corpus_id, exp_name, checkpoint):

    decode_dir = allosaurus_config.tmp_path / 'decode' / corpus_id

    perf = checkpoint.split('/')[-1].split('.')[1]

    # create test directory
    exp_dir = decode_dir / f"{exp_name}-{perf}"
    return exp_dir

def run_prediction(corpus_id, exp_name, checkpoint, batch_size=16, segment_duration=15):

    print("step 1: prediction ---------------------------")

    data_dir = Path('../data/')
    model = read_recognizer(exp_name, checkpoint)

    corpus_path = read_corpus_path(corpus_id)
    lang_id = corpus_path.stem.split('_')[0]
    exp_dir = get_exp_dir(corpus_id, exp_name, checkpoint)
    exp_dir.mkdir(exist_ok=True, parents=True)

    # copy the golden label
    shutil.copyfile(corpus_path / 'text.txt', exp_dir / 'text.txt')

    model.recognize_batch(corpus_path, lang_id, exp_dir, batch_size=batch_size, segment_duration=segment_duration)


def run_tokenization(corpus_id, exp_name, checkpoint):

    print("step 2: tokenization ------------------------")

    corpus_path = read_corpus_path(corpus_id)
    lang_id = corpus_path.stem.split('_')[0]

    exp_dir = get_exp_dir(corpus_id, exp_name, checkpoint)
    exp_dir.mkdir(exist_ok=True, parents=True)

    # copy the golden label
    shutil.copyfile(corpus_path / 'text.txt', exp_dir / 'text.txt')
    w = open(exp_dir / 'phoneme.txt', 'w')

    tokenizer = read_tokenizer(lang_id)
    for line in open(exp_dir / 'text.txt', 'r'):
        fields = line.strip().split()
        utt_id = fields[0]
        sent = ' '.join(fields[1:])
        phonemes = tokenizer.tokenize(sent)
        w.write(utt_id + ' ' + ' '.join(phonemes)+'\n')

    w.close()


def run_eval(corpus_id, exp_name, checkpoint=None):

    print("step 3: eval ------------------------")
    lang_dir = get_exp_dir(corpus_id, exp_name, checkpoint)

    dst = panphon.distance.Distance()

    w = open(lang_dir / "res.txt", "w")

    expect_label = {}

    for line in open(lang_dir / 'phoneme.txt', 'r'):
        fields = line.strip().split()
        utt_id = fields[0]
        expect_label[utt_id] = fields[1:]

    all_cnt = 0
    err_cnt = 0
    dst_cnt = 0

    for line in tqdm(open(lang_dir / 'decode.txt', 'r')):
        fields = line.strip().split()
        utt_id = fields[0]
        pred = fields[1:]

        if utt_id not in expect_label:
            continue

        err_cnt += editdistance.distance(expect_label[utt_id], pred)
        dst_cnt += dst.feature_edit_distance(' '.join(expect_label[utt_id]), ' '.join(pred))
        all_cnt += len(expect_label[utt_id])

    print(corpus_id, ' per: ', f'{err_cnt/all_cnt:.3f} dst: {dst_cnt/all_cnt:.3f}')
    w.write(f"{corpus_id} per : {err_cnt/all_cnt:.3f} dst: {dst_cnt/all_cnt:.3f}\n")
    w.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('a utility to update phone database')
    parser.add_argument('-e', '--exp', help='specify which arch to download. A list of downloadable models are available on Github')
    parser.add_argument('-c', '--checkpoint')
    parser.add_argument('--corpus_id')
    parser.add_argument('-s', '--step', default='1,2,3')
    parser.add_argument('-b', '--batch_size', type=int, default=12)
    parser.add_argument('-d', '--segment_duration', type=int, default=15)

    args = parser.parse_args()
    steps = list(map(int, args.step.split(',')))
    batch_size = int(args.batch_size)
    corpus_id = args.corpus_id
    segment_duration = args.segment_duration

    if args.exp is None:
        exp = Path(args.checkpoint).parent.name
    else:
        exp = args.exp

    if 1 in steps:
        run_prediction(corpus_id, exp, args.checkpoint, batch_size, segment_duration)

    if 2 in steps:
        run_tokenization(corpus_id, exp, args.checkpoint)

    if 3 in steps:
        run_eval(corpus_id, exp, args.checkpoint)
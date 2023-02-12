from phonepiece.iso import normalize_lang_id
from allosaurus.utils.checkpoint_utils import *
from allosaurus.am.model import read_am
from allosaurus.lm.model import read_lm
from allosaurus.pm.model import read_pm
from allosaurus.audio import read_audio
from allosaurus.am.loader import read_audio_loader
from allosaurus.utils.checkpoint_utils import find_topk_models
from allosaurus.am.config import read_exp_config
from allosaurus.am.lang2exp import lang2exp
from allosaurus.bin.download_model import download_model
from allosaurus.utils.model_utils import resolve_model_name, get_all_models
from pathlib import Path
import tqdm
import torch


def read_recognizer(lang_id, alt_model_path=None):

    lang_id = normalize_lang_id(lang_id)

    if lang_id in lang2exp:
        exp_name = lang2exp[lang_id]
    else:
        exp_name = lang2exp['default']

    config = read_exp_config(exp_name)

    # download target model if not exists in local
    download_model(exp_name)

    am = read_am(config.am)
    best_model_path = find_topk_models(exp_name)[0]
    torch_load(am.model, best_model_path)

    if torch.cuda.is_available():
        am.model.cuda()
    else:
        am.device_id = -1

    lm = read_lm(config.lm)
    pm = read_pm(config.pm)

    return Recognizer(am, pm, lm, lang_id)


def read_recognizer_by_exp(exp_name):

    config = read_exp_config(exp_name)

    am = read_am(config.am)
    best_model_path = find_topk_models(exp_name)[0]
    torch_load(am.model, best_model_path)

    if torch.cuda.is_available():
        am.model.cuda()
    else:
        am.device_id = -1

    lm = read_lm(config.lm)
    pm = read_pm(config.pm)

    return Recognizer(am, pm, lm)

class Recognizer:

    def __init__(self, am, pm, lm, lang_id='eng'):

        self.am = am
        self.pm = pm
        self.lm = lm

        self.default_lang_id = lang_id

    def recognize(self, filename, lang_id=None):

        if lang_id is None:
            lang_id = self.default_lang_id

        # load wave audios
        audio = read_audio(filename, self.pm.config.sample_rate)

        # extract feature, add batch dim
        feats = self.pm.compute(audio).unsqueeze(0)
        feat_len = torch.LongTensor([feats.shape[1]])

        if torch.cuda.is_available():
            feats = feats.cuda()
            feat_len = feat_len.cuda()

        # prepare sample
        meta_dict = {'corpus_id': 'test', 'lang_id': lang_id}
        sample_dict = {'feats': (feats, feat_len), 'meta': meta_dict}

        # run inference
        decoded_tokens = self.am.test_step(sample_dict)

        token = self.lm.decode(decoded_tokens[0], lang_id)
        return ' '.join(token)

    def recognize_batch(self, audio_path, lang_id, output_dir):

        audio_loader = read_audio_loader(audio_path, self.pm)

        audio_iterator = iter(audio_loader)
        iteration = len(audio_iterator)

        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        res = []

        # training steps
        for i in tqdm.tqdm(range(iteration)):
            # cur_time = time.time()
            sample = next(audio_iterator)
            sample['meta']['lang_id'] = lang_id

            decoded_tokens = self.am.test_step(sample)

            for utt_id, decoded_token in zip(sample['utt_ids'], decoded_tokens):
                token = self.lm.decode(decoded_token, lang_id)
                res.append((utt_id, token))

        res.sort(key=lambda x: x[0])

        w = open(output_dir / 'decode.txt', 'w')

        for utt_id, token in res:
            w.write(utt_id+' '+' '.join(token)+'\n')
        w.close()

    def get_logits_batch(self, audio_path, lang_id):

        audio_loader = read_audio_loader(audio_path, self.pm)

        audio_iterator = iter(audio_loader)
        iteration = len(audio_iterator)

        logits = []
        tokens = []

        # training steps
        for i in tqdm.tqdm(range(iteration)):
            # cur_time = time.time()
            sample = next(audio_iterator)
            sample['meta']['lang_id'] = lang_id
            sample['meta']['format'] = 'both'

            outputs, decoded_tokens = self.am.test_step(sample)

            for utt_id, decoded_token in zip(sample['utt_ids'], decoded_tokens):
                token = self.lm.decode(decoded_token, lang_id)
                tokens.append((utt_id, token))

            for utt_id, logit in zip(sample['utt_ids'], outputs):
                logits.append((utt_id, logit))

        tokens.sort(key=lambda x: x[0])
        logits.sort(key=lambda x: x[0])

        return logits, tokens

    def get_logits(self, filename, lang_id='ipa'):

        # load wav audio
        audio = read_audio(filename, self.pm.config.sample_rate)

        # extract feature, add batch dim
        feats = self.pm.compute(audio).unsqueeze(0)
        feat_len = torch.LongTensor([feats.shape[1]])

        if torch.cuda.is_available():
            feats = feats.cuda()
            feat_len = feat_len.cuda()

        # prepare sample
        meta_dict = {'corpus_id': 'test', 'lang_id': lang_id, 'format': 'both'}
        sample_dict = {'feats': (feats, feat_len), 'meta': meta_dict}
        output, decoded_tokens = self.am.test_step(sample_dict)
        token = self.lm.decode(decoded_tokens[0], lang_id)

        return output, token
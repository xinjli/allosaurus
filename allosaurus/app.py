from allosaurus.utils.checkpoint_utils import *
from allosaurus.am.model import read_am
from allosaurus.lm.model import read_lm
from allosaurus.pm.model import read_pm
from allosaurus.am.config import read_exp_config
from allosaurus.audio import read_audio, is_audio_file, Audio
from allosaurus.utils.checkpoint_utils import find_topk_models
from allosaurus.am.loader import read_loader, read_audio_loader
from pathlib import Path
import tqdm
import torch
import sys
import numpy as np


def read_recognizer(model_name='latest', checkpoint=None, am_overwrite=None, pm_overwrite=None, lm_overwrite=None):

    model_name, checkpoint = resolve_model_name(model_name, checkpoint)

    # load configs
    config = read_exp_config(model_name)

    am = read_am(config.am, am_overwrite)

    if checkpoint is None:
        checkpoint = find_topk_models(model_name)[0]

    torch_load(am.model, checkpoint)

    if torch.cuda.is_available():
        am.model.cuda()
    else:
        am.device_id = -1

    lm = read_lm(config.lm, lm_overwrite)

    pm = read_pm(config.pm, pm_overwrite)

    return Recognizer(am, pm, lm)


class Recognizer:

    def __init__(self, am, pm, lm):

        self.am = am
        self.pm = pm
        self.lm = lm

    def recognize(self, audio_path, lang_id, output=None, verbose=False, logit=False, batch_size=16, segment_duration=15):

        # recognize a single file
        assert isinstance(audio_path, Audio) or is_audio_file(audio_path)

        audio_loader = read_audio_loader(audio_path, self.pm, batch_size=batch_size, segment_duration=segment_duration)

        audio_iterator = iter(audio_loader)
        iteration = len(audio_iterator)

        utt_infos = []
        utt_logits = []

        # inference steps
        for _ in tqdm.tqdm(range(iteration), disable=(output is not None and output == 'stdout')):

            sample = next(audio_iterator)
            sample['meta']['lang_id'] = lang_id
            sample['meta']['format'] = 'both'

            outputs, decoded_info_lst = self.am.test_step(sample)

            for utt_id, decoded_info in zip(sample['utt_ids'], decoded_info_lst):
                decoded_info = self.lm.decode(decoded_info, lang_id)
                utt_infos.append((utt_id, decoded_info))

            for utt_id, utt_logit in zip(sample['utt_ids'], outputs):
                utt_logits.append((utt_id, utt_logit.cpu().detach().numpy()))

        utt_infos = self.merge_partial_info(utt_infos)
        utt_logits = self.merge_partial_logit(utt_logits)

        assert len(utt_infos) == 1 and len(utt_logits) == 1

        utt_id, decoded_info_lst = utt_infos[0]
        _, lpz = utt_logits[0]

        # write out results
        if output is not None and output == 'stdout':

            # write to stdout
            if verbose:
                for token in decoded_info_lst:
                    print(f"{utt_id} {token['start']:.3f} {token['duration']:.3f} {token['phone']} {token['prob']:.3f}")
            else:
                print(utt_id + ' ' + ' '.join([token['phone'] for token in decoded_info_lst]))

        elif output is not None:

            # write to stdout
            output = Path(output)
            output.mkdir(exist_ok=True, parents=True)

            # write standard output
            w = open(output / 'decode.txt', 'w')
            w.write(utt_id + ' ' + ' '.join([token['phone'] for token in decoded_info_lst]) + '\n')
            w.close()

            # write timestamp, probability
            if verbose:
                # write verbose info
                w = open(output / 'detail.txt', 'w')
                for token in decoded_info_lst:
                    w.write(f"{utt_id} {token['start']:.3f} {token['duration']:.3f} {token['phone']} {token['prob']:.3f}\n")
                w.close()

            # write logits
            if logit:
                print(f"logit shape: {lpz.shape}")
                lpz.dump(output / f'logit.npz')

        else:
            # return results
            if verbose:
                res = decoded_info_lst
            else:
                res = [token['phone'] for token in decoded_info_lst]

            if logit:
                return res, lpz
            else:
                return res


    def recognize_batch(self, audio_path, lang_id, output, verbose=False, batch_size=16, segment_duration=15):

        audio_loader = read_audio_loader(audio_path, self.pm, batch_size=batch_size, segment_duration=segment_duration)

        audio_iterator = iter(audio_loader)
        iteration = len(audio_iterator)

        res = []

        # training steps
        for _ in tqdm.tqdm(range(iteration)):

            sample = next(audio_iterator)
            sample['meta']['lang_id'] = lang_id

            decoded_info_lst = self.am.test_step(sample)

            for utt_id, decoded_info in zip(sample['utt_ids'], decoded_info_lst):
                decoded_info = self.lm.decode(decoded_info, lang_id)
                res.append((utt_id, decoded_info))

        res = self.merge_partial_info(res)

        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True, parents=True)
        w = open(output_dir / 'decode.txt', 'w')

        detail_w = None
        if verbose:
            detail_w = open(output_dir / 'detail.txt', 'w')

        for utt_id, decoded_info_lst in res:

            w.write(utt_id + ' ' + ' '.join([token['phone'] for token in decoded_info_lst]) + '\n')

            # write timestamp, probability
            if verbose:
                for token in decoded_info_lst:
                    w.write(f"{utt_id} {token['start']} {token['duration']} {token['phone']} {token['prob']}\n")

        w.close()
        if detail_w is not None:
            detail_w.close()


    def score_batch(self, corpus_id_or_path, lang_id, output, batch_size=16, num_workers=8):

        print('prep loader')
        loader = read_loader(corpus_id_or_path, self.pm, self.lm, lang_id=lang_id, batch_size=batch_size, num_workers=num_workers)
        print('after prep loader')

        iterator = iter(loader)
        iteration = len(iterator)

        res = []

        # training steps
        for _ in tqdm.tqdm(range(iteration)):

            sample = next(iterator)
            sample['meta']['lang_id'] = lang_id

            loss = self.am.loss_step(sample)

            for utt_id, loss_val in zip(sample['utt_ids'], loss.tolist()):
                res.append((utt_id, loss_val))

        res.sort(key=lambda x: x[0])

        output_dir = Path(output)
        output_dir.mkdir(exist_ok=True, parents=True)
        w = open(output_dir / 'score', 'w')

        for utt_id, loss_val in res:
            w.write(f"{utt_id} {loss_val}\n")

        w.close()

    def merge_partial_logit(self, res):
        res.sort(key=lambda x: x[0])
        merge_res = []

        prev_utt_id = ''
        lpz_lst = []

        for utt_id, lpz in res:
            if len(utt_id) >= 5 and utt_id[-5] == '#' and str.isdigit(utt_id[-4:]):
                cur_utt_id = utt_id[:-5]
            else:
                cur_utt_id = utt_id

            if cur_utt_id != prev_utt_id and prev_utt_id != '':
                lpz = np.concatenate(lpz_lst, axis=0)

                merge_res.append((prev_utt_id, lpz))
                lpz_lst = []

            prev_utt_id = cur_utt_id

            # add 1 additional frame
            lpz = np.concatenate([lpz, lpz[-1][np.newaxis, :]], axis=0)
            lpz_lst.append(lpz)

        if len(lpz_lst) > 0:
            lpz = np.concatenate(lpz_lst, axis=0)
            merge_res.append((prev_utt_id, lpz))

        return merge_res


    def merge_partial_info(self, res):

        res.sort(key=lambda x: x[0])
        merge_res = []

        # no results
        if len(res) == 1 and len(res[0][1]) == 0:
            return res

        prev_utt_id = ''
        out_lst = []

        for utt_id, out in res:
            if len(utt_id) >= 5 and utt_id[-5] == '#' and str.isdigit(utt_id[-4:]):
                cur_utt_id = utt_id[:-5]
            else:
                cur_utt_id = utt_id

            if cur_utt_id != prev_utt_id and prev_utt_id != '':
                merge_res.append((prev_utt_id, out_lst))
                out_lst = []

            prev_utt_id = cur_utt_id
            out_lst.extend(out)

        if len(out_lst) > 0:
            merge_res.append((prev_utt_id, out_lst))

        return merge_res
import numpy as np
from allosaurus.dataset import read_dataset, read_audio_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from allosaurus.utils.tensor import pad_list
import torch


def collate_feat(utt_dict_lst, pm_config):

    feat_lens = [len(utt_dict['feats']) for utt_dict in utt_dict_lst]
    T = max(feat_lens)

    padded_feats = []

    if pm_config.model != 'raw':
        for i, utt_dict in enumerate(utt_dict_lst):
            feat = utt_dict['feats']
            feat_len = feat.shape[0]
            padding_feat_len = T - feat_len
            padded_feats.append(F.pad(feat, pad=(0, 0, 0, padding_feat_len), value=0.0).unsqueeze(0))

        feats = torch.cat(padded_feats, dim=0)
        feat_lens = torch.IntTensor(feat_lens)
    else:

        feat_lst = []
        feat_lens = []

        for i, utt_dict in enumerate(utt_dict_lst):
            feat = utt_dict['feats']
            feat_len = feat.shape[0]
            feat_lst.append(feat)
            feat_lens.append(feat_len)

        feats = pad_list(feat_lst, 0)
        feat_lens = torch.IntTensor(feat_lens)

    return feats, feat_lens


def collate_lang(utt_dict_lst, config):
    B = len(utt_dict_lst)
    T = max(len(utt_dict['langs']) for utt_dict in utt_dict_lst)

    lang = torch.zeros(B, T, dtype=torch.int64)

    lang_lens = []

    for i, utt_dict in enumerate(utt_dict_lst):
        token_ids = utt_dict['langs']

        # target
        target_len = len(token_ids)
        lang[i, :target_len] = token_ids[:target_len]

        # target length
        lang_lens.append(target_len)

    lang_lens = torch.IntTensor(lang_lens)

    return lang, lang_lens


def collate_feats_langs(raw_utt_dict_lst):

    utt_dict_lst = []

    pm_config = raw_utt_dict_lst[0]['pm']
    lm_config = raw_utt_dict_lst[0]['lm']

    # filtering
    for utt_dict in raw_utt_dict_lst:
        lang_length = len(utt_dict['langs'])
        feat_length = len(utt_dict['feats'])

        if lang_length > 500 or lang_length * 2 + 1 >= feat_length:
            continue

        if pm_config.model != 'raw' and feat_length > 2000:
            print("deleting > 2000 ", pm_config)
            continue

        if pm_config.model == 'raw' and feat_length > 160000:
            print("deleting samples > 160000 ", feat_length)
            continue

        assert utt_dict['corpus_id'] == raw_utt_dict_lst[0]['corpus_id'], "corpus id inconsistent!"

        utt_dict_lst.append(utt_dict)

    utt_ids = [utt_dict['utt_id'] for utt_dict in utt_dict_lst]
    batch_dict = {'utt_ids': utt_ids}

    batch_dict['feats'] = collate_feat(utt_dict_lst, pm_config)
    batch_dict['langs'] = collate_lang(utt_dict_lst, lm_config)

    # meta data for this batch
    meta = {}
    meta['lang_id'] = utt_dict_lst[0]['lang_id']
    meta['corpus_id'] = utt_dict_lst[0]['corpus_id']
    batch_dict['meta'] = meta

    return batch_dict

def collate_feats(utt_dict_lst):

    pm_config = utt_dict_lst[0]['pm']

    utt_ids = [utt_dict['utt_id'] for utt_dict in utt_dict_lst]
    batch_dict = {'utt_ids': utt_ids}
    batch_dict['feats'] = collate_feat(utt_dict_lst, pm_config)

    # meta data for this batch
    meta = {}
    batch_dict['meta'] = meta

    return batch_dict


def read_loader(corpus_path, pm_config_or_name, lm_config_or_name, batch_size=12):

    dataset = read_dataset(corpus_path, pm_config_or_name, lm_config_or_name)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_feats_langs, num_workers=8)


def read_audio_loader(corpus_path, pm_config_or_name, batch_size=16, segment_duration=15):

    dataset = read_audio_dataset(corpus_path, pm_config_or_name, segment_duration=segment_duration)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_feats)

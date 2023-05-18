import torch
import numpy as np
from phonepiece.ipa import read_ipa
from torch.nn.utils.rnn import pad_sequence
from scipy.special import softmax
from allosaurus.utils.tensor import make_pad_mask

def feature_tensor_to_embedding(padded_feature, padded_mask, feature_embed):

    # [num_phone, num_feature, num_hidden]
    feature_embed = feature_embed(padded_feature)
    masked_feature = feature_embed.masked_fill(padded_mask, 0.0)

    # [num_phone, num_feature, num_hidden] -> [num_phone, num_hidden] -> [num_hidden, num_phone]
    phone_embed = torch.sum(masked_feature, axis=1).transpose(0, 1)
    return phone_embed


def feature_list_to_tensor(feature_lst):

    tensor_lst = []
    length_lst = []
    for feature_idx in feature_lst:
        tensor_lst.append(torch.LongTensor(feature_idx))
        length_lst.append(len(feature_idx))

    padded_feature = pad_sequence(tensor_lst, batch_first=True)
    padded_mask = make_pad_mask(torch.LongTensor(length_lst)).unsqueeze(-1)
    return padded_feature, padded_mask

def create_feature_list(inventory, pos_only=True):

    phones = inventory.phone.tolist()

    ipa = read_ipa()

    # check format
    assert phones[-1] == '<eos>' and phones[0] == '<blk>'

    feature_lst = []

    # feature for blank
    feature_lst.append([0])

    # process remaining phones
    for phone in phones[1:-1]:

        features = ipa.read_feature(phone)
        feature_idx = []

        for j, label in enumerate(features):

            # +1 to skip the ctc index
            if pos_only:
                if label == 1:
                    feature_idx.append(j+1)
            else:
                if label == 0:
                    continue
                # label: -1, 0, 1 -> 0, 1, 2
                label += 1
                feature_idx.append(3*j + label + 1)

        feature_lst.append(feature_idx)

    return feature_lst


def create_phone_embedding(inventory, feature_embed, pos_only=True):
    """
    build phone embedding from feature embedding
    """

    phones = inventory.phone.tolist()

    ipa = read_ipa()

    # check format
    assert phones[-1] == '<eos>' and phones[0] == '<blk>'

    embed_lst = []

    # embed for blank
    ctc_embed = feature_embed(torch.LongTensor([0]).to(device=feature_embed.weight.device))
    embed_lst.append(ctc_embed)

    # process remaining phones
    for phone in phones[1:-1]:

        features = ipa.read_feature(phone)
        feature_idx = []

        for j, label in enumerate(features):

            # +1 to skip the ctc index
            if pos_only:
                if label == 1:
                    feature_idx.append(j+1)
            else:
                # label: -1, 0, 1 -> 0, 1, 2
                label += 1
                feature_idx.append(3*j + label + 1)


        embed_lst.append(feature_embed(torch.LongTensor(feature_idx).to(device=feature_embed.weight.device)).sum(dim=0).unsqueeze(0))

    phone_embed = (torch.cat(embed_lst, dim=0)).transpose(0,1)
    return phone_embed


def create_allophone_mapping(inventory):

    phonemes = inventory.phoneme.elems
    phones = inventory.phone.elems

    assert phonemes[-1] == '<eos>' and phonemes[0] == '<blk>'
    assert phones[-1] == '<eos>' and phones[0] == '<blk>'

    phonemes = phonemes[:-1]
    phones = phones[:-1]

    mapping = np.zeros([len(phones), len(phonemes)], dtype=np.float32)

    # map <blk> -> <blk>
    mapping[0][0] = 1.0

    for i, phone in enumerate(phones[1:]):
        for phm in inventory.phone2phoneme[phone]:
            phm_id = inventory.phoneme.get_id(phm)
            mapping[i+1][phm_id] = 1.0

    return torch.from_numpy(np.expand_dims(mapping.T, axis=(0,1)))

def create_allophone_distribution(inventory):
    allophone_mat = create_allophone_mapping(inventory)
    allophone_dist = torch.from_numpy(softmax(allophone_mat, axis=-1))
    return allophone_dist
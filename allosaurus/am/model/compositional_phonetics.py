import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from allosaurus.lm.allophone import read_allophone
import panphon
import warnings
from allosaurus.am.utils import make_non_pad_mask, tensor_to_cuda
from allosaurus.am.module.pos import PositionalEncoding
from allosaurus.am.module.transformer import TransformerEncoderLayer
from allosaurus.am.module.conv import ConvFrontEnd
import numpy as np

warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")


def create_feature_table(inventory, ft, embed, config):

    phones = inventory.phone.elems

    # for CTC
    embed_lst = []

    # embed for blank
    ctc_embed = embed(torch.LongTensor([0]).to(device=embed.weight.device))
    embed_lst.append(ctc_embed)

    for phone in phones[1:]:
        if phone == 'g':
            phone = 'ɡ'
        elif phone == 'gʷ':
            phone = 'ɡʷ'
        elif phone == 'gː':
            phone = 'ɡ'
        elif phone == 'ә':
            phone = 'ə'
        elif phone == 'I':
            phone = 'ɪ'

        fts = ft.word_fts(phone)
        if len(fts) == 0:
            # 1 for default embedding
            embed_lst.append(embed(torch.LongTensor([1]).to(device=embed.weight.device)))
        else:
            phone_index_lst = []

            attributes = fts[0].numeric()

            if len(attributes) == 24:
                attributes = attributes[:-2]

            for j, label in enumerate(fts[0].numeric()):

                if label == 1:
                    phone_index_lst.append(j+2)

            embed_lst.append(embed(torch.LongTensor(phone_index_lst).to(device=embed.weight.device)).sum(dim=0).unsqueeze(0))


    mat = (torch.cat(embed_lst, dim=0)).transpose(0,1)
    return mat

def allophone_layer(allophone):

    phoneme = allophone.phoneme

    phone_set = allophone.phone.elems
    phoneme_set = phoneme.elems

    mapping = np.zeros([len(phone_set), len(phoneme_set)], dtype=np.float32)

    # map <blk> -> <blk>
    mapping[0][0] = 1.0

    for i, phone in enumerate(phone_set):
        for phm in allophone.phone2phoneme[phone]:
            phm_id = phoneme.get_id(phm)
            mapping[i][phm_id] = 1.0

    return torch.from_numpy(np.expand_dims(mapping.T, axis=(0,1)))

class CompositionalPhoneticsModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size

        #self.lang_size_dict = config.lang_size_dict
        self.lang_output_size = dict()

        if config.posonly:
            self.embed = nn.Embedding(32, self.hidden_size)
        else:
            self.embed = nn.Embedding(64, self.hidden_size)


        self.phone_tensor = None

        self.feature2phone = nn.ParameterDict()
        self.phone2phoneme = nn.ParameterDict()
        self.phone_mask = nn.ParameterDict()

        self.phone2phoneme_params = dict()

        # self.output_layer = nn.Linear(self.hidden_size*2, 40)

        self.ft = panphon.FeatureTable()

        for lang_id in config.langs:
            #print("preparing ", lang_id)

            allophone = read_allophone(self.config.model_path, lang_id)

            #self.phone2phoneme_params[lang_id] = nn.Parameter(torch.from_numpy(self.phone[lang_id].get_phone2phoneme_mapping(lang_id)).float(), requires_grad=False)
            self.feature2phone[lang_id] = nn.Parameter(create_feature_table(allophone, self.ft, self.embed, self.config))
            # self.feature2phone[lang_id] = nn.Parameter(torch.from_numpy(create_feature_table(lang_id, self.ft, self.embed)), requires_grad=False)
            allophone_map = allophone_layer(allophone)
            self.phone2phoneme[lang_id] = nn.Parameter(allophone_map, requires_grad=False)
            self.phone_mask[lang_id] = nn.Parameter(~allophone_map.bool(), requires_grad=False)


        # transformers
        self.normalize_before = config.normalize_before
        self.relative_positional = config.relative_positional

        self.frontend = ConvFrontEnd(input_size=config.input_size,
                                     output_size=config.hidden_size,
                                     in_channel = config.in_channel,
                                     mid_channel = config.mid_channel,
                                     out_channel = config.out_channel,
                                     kernel_size = config.kernel_size,
                                     stride = config.stride,
                                     dropout = 0.0,
                                     act_func_type = config.act_func_type,
                                     front_end_layer_norm = config.front_end_layer_norm)


        self.pos_emb = PositionalEncoding(config.hidden_size, config.pos_dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(config) for _ in range(config.block_size)
        ])

        if self.normalize_before:
            self.norm = nn.LayerNorm(config.hidden_size)


        self.logsoftmax = nn.LogSoftmax(dim=2)

    def forward(self, input_tensor, input_lengths, meta=None):
        """

        :param input: an Tensor with shape (B,T,H)
        :lengths: a list of length of input_tensor, if None then no padding
        :meta: dictionary containing meta information (should contain lang_id in this case
        :return:
        """

        #if utt_ids:
            #print("utt_ids {} \n target_tensor {}".format(' '.join(utt_ids), target_tensor))
            #print("input_lengths {}".format(str(input_lengths)))
            #print("target_tensor {}".format(target_tensor))
            #print("target_lengths {}".format(target_lengths))



        if meta['lang_id'] not in self.feature2phone:
            lang_id = meta['lang_id']

            #print("Creating new layer for ", lang_id)
            allophone = read_allophone(self.config.model_path, lang_id)
            self.feature2phone[lang_id] = nn.Parameter(tensor_to_cuda(create_feature_table(allophone, self.ft, self.embed, self.config), device_id=meta['device_id']))

            allophone_map = allophone_layer(allophone)
            self.phone2phoneme[lang_id] = nn.Parameter(tensor_to_cuda(allophone_map, device_id=meta['device_id']), requires_grad=False)
            self.phone_mask[lang_id] = nn.Parameter(tensor_to_cuda(~allophone_map.bool(), device_id=meta['device_id']), requires_grad=False)
            #self.allophone_layer_dict[lang_id] = nn.Parameter(torch.from_numpy(create_feature_table(lang_id, self.ft)).cuda(), requires_grad=False)


        mask = make_non_pad_mask(input_lengths)

        input_tensor, mask = self.frontend(input_tensor, mask)


        if self.relative_positional:
            enc_output = input_tensor
            # [1, 2T - 1]
            position = torch.arange(-(input_tensor.size(1)-1), input_tensor.size(1), device=input_tensor.device).reshape(1, -1)
            pos = self.pos_emb._embedding_from_positions(position)
        else:
            enc_output, pos = self.pos_emb(input_tensor)

        # enc_output.masked_fill_(~mask.unsqueeze(2), 0.0)

        attn_weights = {}
        for i, block in enumerate(self.blocks):
            enc_output, _, attn_weight = block.inference(enc_output, mask.unsqueeze(1), pos)
            attn_weights['enc_block_%d' % i] = attn_weight

        if self.normalize_before:
            enc_output = self.norm(enc_output)

        # return enc_output, mask, attn_weights

        # output_tensor = self.logsoftmax(self.output_layer(enc_output))

        output_length = torch.sum(mask, dim=1)

        # (T,B,F) -> (T,B,F) x (F,P) -> (T,B,P)
        phone_tensor = torch.matmul(enc_output, self.feature2phone[meta['lang_id']])

        phone_tensor /= torch.sqrt(torch.tensor(self.hidden_size*1.0))

        #phone_tensor = self.phone_softmax(phone_tensor)

        #print("feature2phone:", self.feature2phone[meta['lang_id']])
        #print("phone_tensor: ", phone_tensor)

        # (T,B,1,P) x (1,1,Q,P) = (T,B,Q,P)
        allophone_tensor = torch.mul(phone_tensor.unsqueeze(2), self.phone2phoneme[meta['lang_id']])

        allophone_tensor = allophone_tensor.masked_fill(self.phone_mask[meta['lang_id']], -np.inf)
        #print("allophone_tensor: ", allophone_tensor)

        #  (T,B,Q,P) -> (T,B,Q)
        phoneme_tensor = torch.max(allophone_tensor, axis=-1)[0]

        #print("phoneme_tensor: ", phoneme_tensor)

        # output layer (T,B,H)
        output_tensor = self.logsoftmax(phoneme_tensor)

        #output_tensor = self.logsoftmax(self.output_layer(output_tensor))


        #print("in - input_tensor ", input_tensor.shape)
        #print("in - output_tensor ", output_tensor.shape)
        #print("in - input lengths", input_lengths.shape)
        #print("in - target tensor", target_tensor.shape)
        #print("in - target length", target_lengths.shape)

        # return (B,T,H) for gathering
        return output_tensor
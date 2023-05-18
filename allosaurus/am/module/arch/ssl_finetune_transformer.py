import torch
import torch.nn as nn
from phonepiece.inventory import read_inventory
from allosaurus.am.module.utils.phonetics import *
from allosaurus.am.module.encoder.transformer import TransformerEncoderLayer
import warnings
from allosaurus.utils.tensor import make_non_pad_mask, pad_list
import numpy as np
from allosaurus.am.module.frontend.ssl import read_ssl_frontend
from allosaurus.am.module.block.pos import PositionalEncoding
from allosaurus.am.module.utils.register import register_arch

#warnings.filterwarnings("ignore", message="Setting attributes on ParameterDict is not supported.")

@register_arch
class SSLFinetuneTransformer(nn.Module):

    type_ = 'ssl_finetune_transformer'

    def __init__(self, config):
        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size

        self.lang_output_size = dict()

        self.phone_tensor = None

        # prepare SSL frontend
        self.frontend = read_ssl_frontend(config)

        # prepare phonetics layers
        if self.config.pos_only:
            self.embed = nn.Embedding(32, self.hidden_size)
        else:
            self.embed = nn.Embedding(96, self.hidden_size)

        self.feature2phone = nn.ParameterDict()
        self.phone2phoneme = nn.ParameterDict()
        self.phone_mask = nn.ParameterDict()

        self.phone2phoneme_params = dict()

        # self.output_layer = nn.Linear(self.hidden_size*2, 40)
        self.blocks = nn.ModuleList([
           TransformerEncoderLayer(config) for _ in range(config.block_size)
        ])

        for lang_id in config.langs:
            self.prep_language_layer(lang_id, 'cpu')

        #print(config)
        if self.config.rank is None:
            self.config.rank = 0

        self.logsoftmax = nn.LogSoftmax(dim=2)

    def prep_language_layer(self, lang_id, device=None):

        if str(lang_id) not in self.feature2phone:
            #print("preparing ", lang_id)

            inventory = read_inventory(lang_id)
            lang_id = str(lang_id)

            self.feature2phone[lang_id] = nn.Parameter(create_phone_embedding(inventory, self.embed, self.config.pos_only)).to(device)
            allophone_map = create_allophone_mapping(inventory).to(device)
            self.phone2phoneme[lang_id] = nn.Parameter(allophone_map, requires_grad=False)
            self.phone_mask[lang_id] = nn.Parameter(~allophone_map.bool(), requires_grad=False)


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

        self.prep_language_layer(meta['lang_id'], input_tensor.device)

        lang_id = str(meta['lang_id'])

        feats, mask = self.frontend.forward(input_tensor, input_lengths)

        enc_output = feats
        #print("feats: ", enc_output)
        #print("feats shape: ", enc_output.shape)

        for i, block in enumerate(self.blocks):
            enc_output, _ = block(enc_output, mask.unsqueeze(1))

        output_length = torch.sum(mask, dim=1)

        # (T,B,F) -> (T,B,F) x (F,P) -> (T,B,P)
        # print(self.feature2phone[meta['lang_id']].shape)

        phone_tensor = torch.matmul(enc_output, self.feature2phone[lang_id])
        #print("feature2phone:", self.feature2phone[meta['lang_id']])
        #print("phone_tensor: ", phone_tensor)
        #print("phone_tensor_shape: ", phone_tensor.shape)

        phone_tensor /= torch.sqrt(torch.tensor(self.hidden_size*1.0))

        #phone_tensor = self.phone_softmax(phone_tensor)

        #print("feature2phone:", self.feature2phone[meta['lang_id']])
        #print("after norm phone_tensor: ", phone_tensor)

        # (T,B,1,P) x (1,1,Q,P) = (T,B,Q,P)
        # print(phone_tensor.shape, self.phone2phoneme[meta['lang_id']].shape)

        allophone_tensor = torch.mul(phone_tensor.unsqueeze(2), self.phone2phoneme[lang_id])

        allophone_tensor = allophone_tensor.masked_fill(self.phone_mask[lang_id], -np.inf)
        #print("allophone_tensor: ", allophone_tensor)

        #  (T,B,Q,P) -> (T,B,Q)
        phoneme_tensor = torch.max(allophone_tensor, axis=-1)[0]

        #print("phoneme_tensor: ", phoneme_tensor)

        # output layer (T,B,H)
        output_tensor = self.logsoftmax(phoneme_tensor)
        #print("softmax ", output_tensor)

        #output_tensor = self.logsoftmax(self.output_layer(output_tensor))

        #print("in - input_tensor ", input_tensor.shape)
        #print("in - output_tensor ", output_tensor.shape)
        #print("in - input lengths", input_lengths.shape)
        #print("in - target tensor", target_tensor.shape)
        #print("in - target length", target_lengths.shape)

        result = {
            'output': output_tensor,
            'output_length': output_length,
        }

        # return (B,T,H) for gathering
        return result
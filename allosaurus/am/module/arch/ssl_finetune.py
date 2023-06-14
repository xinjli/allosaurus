import torch
import torch.nn as nn
from allosaurus.am.module.frontend.ssl import read_ssl_frontend
from allosaurus.am.module.utils.register import register_arch
from allosaurus.am.module.encoder.transformer import TransformerEncoderLayer
from allosaurus.am.module.utils.phonetics import create_remap_matrix
from phonepiece.inventory import read_inventory


@register_arch
class SSLFinetune(nn.Module):

    type_ = 'ssl_finetune'

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.hidden_size = config.hidden_size

        self.lang_output_size = dict()

        self.phone_tensor = None

        # prepare SSL frontend
        self.frontend = read_ssl_frontend(config)

        self.logsoftmax = nn.LogSoftmax(dim=2)

        self.lang2linear = nn.ParameterDict()

        self.default_lang = config.langs[0]

        for lang in config.langs:
            self.prep_language_layer(lang, self.config.device)

        self.remap_layer = {}

    def prep_language_layer(self, lang_id, device=None):

        if str(lang_id) not in self.lang2linear:

            inventory = read_inventory(lang_id)
            num_phoneme = len(inventory.phoneme.elems) - 1

            lang_id = str(lang_id)
            print("preparing ", lang_id, ' pos_only ', self.config.pos_only, ' inventory ', str(inventory))
            self.lang2linear[lang_id] = nn.Linear(1024, num_phoneme).to(device)

    def prep_remap_language_layer(self, lang_id, device=None):

        if lang_id in self.lang2linear or lang_id in self.remap_layer:
            return

        remap_matrix = create_remap_matrix(self.default_lang, lang_id).to(device)
        print("create remapping matrix from {} to {}".format(self.default_lang, lang_id))
        print("shape: ", remap_matrix.shape)
        self.remap_layer[lang_id] = remap_matrix

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

        lang_id = str(meta['lang_id'])

        self.prep_remap_language_layer(lang_id, input_tensor.device)

        feats, mask = self.frontend.forward(input_tensor, input_lengths)

        output_length = torch.sum(mask, dim=1)

        if lang_id in self.lang2linear:
            predicted = self.lang2linear[lang_id](feats)
        else:
            predicted = self.lang2linear[self.default_lang](feats)
            predicted = torch.matmul(predicted, self.remap_layer[lang_id])

        output_tensor = self.logsoftmax(predicted)

        result = {
            'output': output_tensor,
            'output_length': output_length,
        }

        # return (B,T,H) for gathering
        return result
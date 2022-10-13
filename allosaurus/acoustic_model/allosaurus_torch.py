import torch
import torch.nn as nn

class AllosaurusTorchModel(nn.Module):

    def __init__(self, config):
        super(AllosaurusTorchModel, self).__init__()

        self.hidden_size = config.hidden_size
        self.layer_size = config.layer_size
        self.proj_size = config.proj_size

        # decide input feature size
        if config.feat_size == -1:
            corpus_feat_size_dict = list(config.feat_size_dict.values())[0]
            self.feat_size = list(corpus_feat_size_dict.values())[0]

        else:
            self.feat_size = config.feat_size

        assert hasattr(config, 'lang_size_dict'), " config should has the lang_size_dict property"

        self.lang_size_dict = config.lang_size_dict
        self.lang_output_size = dict()

        self.phone_size = config.phone_size

        self.config = config

        self.blstm_layer = nn.LSTM(self.feat_size, self.hidden_size, num_layers=self.layer_size, bidirectional=True)

        self.phone_layer = nn.Linear(self.hidden_size*2, self.phone_size)

        self.phone_tensor = None

    @staticmethod
    def add_args(parser):
        parser.add_argument('--feat_size',   type=int, default=-1, help='input size in the blstm model. if -1, then it is determined automatically by loader')
        parser.add_argument('--hidden_size', type=int, default=320, help='hidden size in the blstm model')
        parser.add_argument('--lang_size',   type=int, default=-1, help='output size in the blstm model, if -1, then it is determined automatically by loader')
        parser.add_argument('--proj_size',   type=int, default=0, help='projection')
        parser.add_argument('--layer_size',  type=int, default=5, help='layer size in the blstm model')
        parser.add_argument('--l2', type=float, default=0.0, help='regularization')
        parser.add_argument('--loss', type=str, default='ctc', help='ctc/warp_ctc/e2e')
        parser.add_argument('--debug_model', type=str, default=False, help='print tensor info for debugging')


    def forward(self, input_tensor, input_lengths, return_lstm=False, return_both=False, meta=None):
        """

        :param input: an Tensor with shape (B,T,H)
        :lengths: a list of length of input_tensor, if None then no padding
        :meta: dictionary containing meta information (should contain lang_id in this case
        :return_lstm: [list containing the output_embeddings and their respective lengths]
        :return_both: tuple containing (a list containing the output_embeddings and their respective lengths and the ouptut of phone layer)
        :return:
        """

        #if utt_ids:
            #print("utt_ids {} \n target_tensor {}".format(' '.join(utt_ids), target_tensor))
            #print("input_lengths {}".format(str(input_lengths)))
            #print("target_tensor {}".format(target_tensor))
            #print("target_lengths {}".format(target_lengths))


        # (B,T,H) -> (T,B,H)
        input_tensor = input_tensor.transpose(0, 1).float()

        # extract lengths
        if input_lengths is None:
            input_lengths = torch.LongTensor([input_tensor.shape[0]]*input_tensor.shape[1])

        # keep the max length for padding
        total_length = input_tensor.size(0)

        #if self.config.loss == 'warp_ctc':
        #target_tensor = torch.cat([target_tensor[idx,:index] for idx, index in enumerate(target_lengths)])

        #if lengths.dim() == 2:
        #    lengths = lengths.squeeze()

        # build each layer

        # (T,B,H) -> PackSequence
        pack_sequence = nn.utils.rnn.pack_padded_sequence(input_tensor, input_lengths.cpu())

        # PackSequence -> (PackSequence, States)
        self.blstm_layer.flatten_parameters()

        hidden_pack_sequence, _ = self.blstm_layer(pack_sequence)

        # PackSequence -> (T,B,2H), lengths
        output_tensor, _ = nn.utils.rnn.pad_packed_sequence(hidden_pack_sequence, total_length=total_length)

        # (T,B,2H) -> (T,B,P)
        phone_tensor = self.phone_layer(output_tensor)

        #added the return_lstm argument
        if return_lstm: 
            return [output_tensor.cpu(),input_lengths.cpu()]
        if return_both:
            return [(output_tensor.cpu(),input_lengths.cpu()), phone_tensor.transpose(0,1)]
        
        # return (B,T,H) for gathering
        return phone_tensor.transpose(0,1) 
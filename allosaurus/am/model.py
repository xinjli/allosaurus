from allosaurus.am.config import read_config
from allosaurus.utils.tensor import *
from allosaurus.am.module.arch import *
from allosaurus.am.module.utils.register import arch_types
from itertools import groupby
import editdistance
from collections import defaultdict
import torch.nn as nn
import torch.cuda


def read_am(config_or_name, overwrite_config=None):

    if isinstance(config_or_name, str):
        config = read_config('am/'+config_or_name, overwrite_config)
    else:
        config = config_or_name

    model = None

    for arch_cls in arch_types:
        if arch_cls.type_ == config.model:
            model = arch_cls(config)

    if model is None:
        reporter.critical(f"am arch {config.model} not available from {list(arch_cls.type_ for arch_cls in arch_types)}")
        exit(1)

    if config.criterion == 'ctc':
        from allosaurus.am.module.criterion.ctc_loss import CTCCriterion
        criterion = CTCCriterion(config)
    else:
        reporter.critical("Wrong am criterion ", config.criterion)
        exit(1)

    am = AcousticModel(model, criterion, config)
    return am


class AcousticModel:

    def __init__(self, model, criterion, config):

        self.model = model
        self.criterion = criterion

        self.config = config

        if "rank" in self.config:
            self.device_id = self.config.rank
        elif torch.cuda.is_available():
            self.device_id = 0
        else:
            self.device_id = -1

    def cuda(self):
        self.model = self.model.cuda()
        self.criterion = self.criterion.cuda()

        return self

    def distribute(self):
        self.model = self.model.to(self.config.rank)
        self.device_id = self.config.rank
        self.criterion = self.criterion.to(self.config.rank)
        self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.config.rank], find_unused_parameters=True)
        torch.cuda.set_device(self.config.rank)
        return self

    def train_step(self, sample: dict):

        # print(sample)

        self.model.train()

        # get batch and prepare it into pytorch format
        lang, lang_length = tensor_to_cuda(sample["langs"], self.device_id)
        feat, feat_length = tensor_to_cuda(sample["feats"], self.device_id)

        meta = sample['meta']

        result = self.model(feat, feat_length, meta)

        output = result['output']
        output_length = result['output_length']

        # print("output ", output, "shape ", output.shape)
        # print("output_length ", output_length)
        # print("lang ", lang)
        # print("lang_length ", lang_length)
        ter, total_token_size,target_tokens, decoded_tokens = self.eval_ter(output, output_length, sample)

        loss = self.criterion(output, output_length, lang, lang_length)


        total_loss = float(loss)

        # print(feat, feat_length)
        # print(total_loss, total_token_size)

        ave_loss = loss / total_token_size

        aux_loss = 0

        if 'loss' in result:
            ave_loss += result['loss']
            aux_loss = float(result['loss'])

        ave_loss.backward()

        #print(output_length / lang_length)
        #print(output.shape[1] / lang.shape[1])
        # if torch.any(torch.isnan(self.arch.frontend.conv1.conv_layer.weight.grad)):
        #     print("wrong !!!!!")
        #     print(sample['meta'])
        #     print(sample['utt_ids'])
        #     print("feat ", feat)
        #     print("feat shape ", feat.shape)
        #     print("lang ", lang)
        #     print("lang shape ", lang.shape)
        #     print("max lang ", torch.max(lang, axis=1))
        #     print("output ", output)
        #     print("output shape ", output.shape)
        #     print("max output ", torch.max(output))
        #     exit(1)
        #if self.config.debug_task:
        # debug_model(self.arch)

        corpus_id = sample['meta']['corpus_id']

        step_report = {
            'corpus_id': corpus_id,
            'total_ter': ter,
            'total_token_size': total_token_size,
            'total_loss': total_loss,
            'aux_loss': aux_loss,
            'average_ter': ter / total_token_size,
            'average_loss': total_loss / total_token_size,
            'output': output,
            'output_length': output_length,
            'lang': lang,
            'lang_length': lang_length,
            'ref_tokens': target_tokens,
            'hyp_tokens': decoded_tokens
        }

        return step_report

    def validate_step(self, sample: dict):

        self.model.eval()

        # get batch and prepare it into pytorch format
        lang, lang_length = tensor_to_cuda(sample["langs"], self.device_id)
        feat, feat_length = tensor_to_cuda(sample["feats"], self.device_id)

        meta = sample['meta']

        result = self.model(feat, feat_length, meta)

        output = result['output']
        output_length = result['output_length']

        loss = self.criterion(output, output_length, lang, lang_length)

        ter, total_token_size, target_tokens, decoded_tokens = self.eval_ter(output, output_length, sample)

        total_loss = float(loss)

        if self.config.debug_task:
            debug_model(self.model)

        corpus_id = sample['meta']['corpus_id']

        step_report = {
            'corpus_id': corpus_id,
            'total_ter': ter,
            'total_token_size': total_token_size,
            'total_loss': total_loss,
            'aux_loss': 0,
            'average_ter': ter / total_token_size,
            'average_loss': total_loss / total_token_size
        }

        return step_report

    def loss_step(self, sample: dict):

        self.model.eval()

        # get batch and prepare it into pytorch format
        lang, lang_length = tensor_to_cuda(sample["langs"], self.device_id)
        feat, feat_length = tensor_to_cuda(sample["feats"], self.device_id)

        meta = sample['meta']

        result = self.model(feat, feat_length, meta)

        output = result['output']
        output_length = result['output_length']

        loss = self.criterion.non_reduction_forward(output, output_length, lang, lang_length)
        loss = (loss / lang_length).cpu().detach().numpy()

        return loss


    def test_step(self, sample: dict):

        self.model.eval()

        # get batch and prepare it into pytorch format
        feat, feat_length = tensor_to_cuda(sample["feats"], self.device_id)

        meta = sample['meta']

        if 'format' in meta:
            format = meta['format']
        else:
            format = 'token'

        result = self.model(feat, feat_length, meta)

        output = result['output'].detach()
        output_length = result['output_length']

        logits = move_to_ndarray(output)
        logits_length = move_to_ndarray(output_length)

        assert len(logits) == len(logits_length)
        logit_lst = []

        for ii in range(len(logits)):
            logit = logits[ii][:logits_length[ii]]
            logit_lst.append(logit)

        emit = 1.0
        if 'emit' in sample:
            emit = sample['emit']

        timestamp = False
        if 'timestamp' in sample:
            timestamp = sample['timestamp']

        topk = 1
        if 'topk' in sample:
            topk = sample['topk']

        if format == 'logit':
            return logit_lst
        elif format == 'both':
            decoded_info = self.decode(logit_lst, topk, emit)
            return output, decoded_info

        else:
            decoded_info = self.decode(logit_lst, topk, emit)

            return decoded_info

    def reduce_report(self, reports):

        corpus_reports_dict = defaultdict(list)

        # group by corpus_id
        for report in reports:
            corpus_reports_dict[report['corpus_id']].append(report)

        corpus_report = dict()

        # do analysis per corpus
        for corpus_id, corpus_reports in corpus_reports_dict.items():
            corpus_dict = dict()
            corpus_dict['total_token_size'] = sum([report['total_token_size'] for report in corpus_reports])
            corpus_dict['total_ter'] = sum([report['total_ter'] for report in corpus_reports])
            corpus_dict['total_loss'] = sum([report['total_loss'] for report in corpus_reports])
            corpus_dict['aux_loss'] = sum([report['aux_loss'] for report in corpus_reports])
            corpus_dict['average_ter'] = corpus_dict['total_ter']*1.0 / corpus_dict['total_token_size']
            corpus_dict['average_loss'] = corpus_dict['total_loss']*1.0 / corpus_dict['total_token_size']

            corpus_report[corpus_id] = corpus_dict

        # do analysis totally
        total_report = dict()
        total_report['total_token_size'] = sum([report['total_token_size'] for report in reports])
        total_report['total_ter'] = sum([report['total_ter'] for report in reports])
        total_report['total_loss'] = sum([report['total_loss'] for report in reports])
        total_report['aux_loss'] = sum([report['aux_loss'] for report in reports])
        total_report['average_ter'] = total_report['total_ter']*1.0 / total_report['total_token_size']
        total_report['average_loss'] = total_report['total_loss']*1.0 / total_report['total_token_size']

        return total_report, corpus_report

    def eval_ter(self, output, output_length, sample):
        """
        compute SUM of ter in this batch

        :param batch_logits: (B,T,Token_size)
        :param batch_target:  (B, max_label)
        :param batch_target_len: [B]
        :return:
        """

        # print("shape is ", batch_logits.shape)

        error_cnt = 0.0
        total_cnt = 0.0

        logits = move_to_ndarray(output)
        logits_length = move_to_ndarray(output_length)

        targets, targets_length = sample["langs"]
        target_tokens = []
        decoded_tokens = []

        for i in range(len(targets_length)):
            target = targets[i, :targets_length[i]].tolist()
            logit  = logits[i][:logits_length[i]]

            raw_token = [x[0] for x in groupby(np.argmax(logit, axis=1))]
            decoded_token = list(filter(lambda a: a != 0, raw_token))

            target_tokens.append(target)
            decoded_tokens.append(decoded_token)
            #print('target ', target)
            #print('decoded_token ', decoded_token)

            error = editdistance.distance(target, decoded_token)

            error_cnt += error
            total_cnt += targets_length[i]

        return error_cnt, total_cnt, target_tokens, decoded_tokens

    # def decode(self, output, output_length, meta):
    #
    #     logits = move_to_ndarray(output)
    #     logits_length = move_to_ndarray(output_length)
    #
    #     assert len(logits) == len(logits_length)
    #
    #     decoded_tokens = []
    #
    #     for i in range(len(logits)):
    #
    #         logit  = logits[i][:logits_length[i]]
    #
    #         raw_token = [x[0] for x in groupby(np.argmax(logit, axis=1))]
    #         decoded_token = list(filter(lambda a: a != 0, raw_token))
    #         decoded_tokens.append(decoded_token)
    #
    #     return decoded_tokens


    def decode(self, logit_lst, topk=1, emit=1.0):

        decoded_lst = []

        for ii in range(len(logit_lst)):

            logit = logit_lst[ii]
            emit_frame_idx = []

            cur_max_arg = -1

            # find all emitting frames
            for i in range(len(logit)):

                frame = logit[i]
                frame[0] /= emit

                arg_max = np.argmax(frame)

                # this is an emitting frame
                if arg_max != cur_max_arg and arg_max != 0:
                    emit_frame_idx.append(i)
                    cur_max_arg = arg_max

            # decode all emitting frames
            decoded_seq = []

            for i, idx in enumerate(emit_frame_idx):
                frame = logit[idx]

                if i == len(emit_frame_idx) - 1:
                    next_idx = len(logit)-1
                else:
                    next_idx = emit_frame_idx[i+1]

                exp_prob = np.exp(frame - np.max(frame))
                probs = exp_prob / exp_prob.sum()

                top_phones = frame.argsort()[-topk:][::-1]
                top_probs = sorted(probs)[-topk:][::-1]

                start_timestamp = self.config.window_shift * idx
                end_timestamp = self.config.window_shift * next_idx
                duration = min(0.2, end_timestamp - start_timestamp)

                info = {'start': self.config.window_shift * idx,
                        'duration': duration,
                        'phone_id': top_phones[0],
                        'prob': top_probs[0],
                        'alternative_ids': top_phones[:topk].tolist(),
                        'alternative_probs': top_probs[:topk]}

                decoded_seq.append(info)

            decoded_lst.append(decoded_seq)

            #raw_token = [x[0] for x in groupby(np.argmax(logit, axis=1))]
            #decoded_token = list(filter(lambda a: a != 0, raw_token))
            #decoded_tokens.append(decoded_token)

        return decoded_lst
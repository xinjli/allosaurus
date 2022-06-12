from allosaurus.lm.inventory import *
from allosaurus.lm.allophone import read_allophone
from pathlib import Path
from itertools import groupby
import numpy as np


class PhoneDecoder:

    def __init__(self, model_path, inference_config):
        """
        This class is an util for decode both phones and words

        :param model_path:
        """

        # lm model path
        self.model_path = Path(model_path)

        self.config = inference_config

        # setup language id
        if inference_config.lang is not None:
            self.lang = inference_config.lang
        else:
            self.lang = 'eng'

        # create inventory
        if self.config.model != 'interspeech21':
            self.inventory = Inventory(model_path, inference_config)
            self.unit = self.inventory.unit
            self.config.offset = 0
        else:
            self.allophone = read_allophone(model_path, self.lang)
            self.config.window_shift *= 4
            self.config.window_size = 0.055
            self.config.offset = 0.035

    def is_available(self, lang_id):

        if self.inventory is not None:
            return self.inventory.is_available(lang_id)
        else:
            return self.model_path / 'inventory' / lang_id

    def compute(self, logits, lang_id=None, topk=1, emit=1.0, timestamp=False, phoneme=False):
        """
        decode phones from logits

        :param logits: numpy array of logits
        :param emit: blank factor
        :return:
        """

        # In the original allosaurus model
        # we apply mask if lang_id specified, this is to restrict the output phones to the desired phone subset
        if self.config.model != 'interspeech21':
            mask = self.inventory.get_mask(lang_id, approximation=self.config.approximate)
            logits = mask.mask_logits(logits)
        else:
            if lang_id is not None and lang_id != self.lang:
                self.lang = lang_id
                self.allophone = read_allophone(self.model_path, self.lang)

            mask = self.allophone.phoneme

        emit_frame_idx = []

        cur_max_arg = -1

        # find all emitting frames
        for i in range(len(logits)):

            logit = logits[i]
            logit[0] /= emit

            arg_max = np.argmax(logit)

            # this is an emitting frame
            if arg_max != cur_max_arg and arg_max != 0:
                emit_frame_idx.append(i)
                cur_max_arg = arg_max

        # decode all emitting frames
        decoded_seq = []
        for idx in emit_frame_idx:
            logit = logits[idx]
            exp_prob = np.exp(logit - np.max(logit))
            probs = exp_prob / exp_prob.sum()

            top_phones = logit.argsort()[-topk:][::-1]
            top_probs = sorted(probs)[-topk:][::-1]

            stamp = f"{self.config.offset + self.config.window_shift*idx:.3f} {self.config.window_size:.3f} "

            if topk == 1:

                phones_str = ' '.join(mask.get_units(top_phones))
                if timestamp:
                    phones_str = stamp + phones_str

                decoded_seq.append(phones_str)
            else:
                phone_prob_lst = [f"{phone} ({prob:.3f})" for phone, prob in zip(mask.get_units(top_phones), top_probs)]
                phones_str = ' '.join(phone_prob_lst)

                if timestamp:
                    phones_str = stamp + phones_str

                decoded_seq.append(phones_str)

            # in the interspeech21 model, we support output at the phoneme level
            if phoneme and self.config.model == 'interspeech21':
                decoded_seq = [self.allophone.phone2phoneme[phone][0] for phone in decoded_seq]

        if timestamp:
            phones = '\n'.join(decoded_seq)
        elif topk == 1:
            phones = ' '.join(decoded_seq)
        else:
            phones = ' | '.join(decoded_seq)

        return phones
from allosaurus.lm.inventory import *
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

        # create inventory
        self.inventory = Inventory(model_path, inference_config)

        self.unit = self.inventory.unit

    def compute(self, logits, lang_id=None, topk=1, emit=1.0, timestamp=False):
        """
        decode phones from logits

        :param logits: numpy array of logits
        :param emit: blank factor
        :return:
        """

        # apply mask if lang_id specified, this is to restrict the output phones to the desired phone subset

        mask = self.inventory.get_mask(lang_id, approximation=self.config.approximate)

        logits = mask.mask_logits(logits)

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

            stamp = f"{self.config.window_shift*idx:.3f} {self.config.window_size:.3f} "

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

        if timestamp:
            phones = '\n'.join(decoded_seq)
        elif topk == 1:
            phones = ' '.join(decoded_seq)
        else:
            phones = ' | '.join(decoded_seq)

        return phones
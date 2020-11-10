from allosaurus.lm.inventory import *
from pathlib import Path
from itertools import groupby

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
        self.inventory = Inventory(model_path)

        self.unit = self.inventory.unit

    def compute(self, logits, lang_id=None, topk=1, blank_factor=1.0):
        """
        decode phones from logits

        :param logits: numpy array of logits
        :param blank_factor:
        :return:
        """

        # apply mask if lang_id specified, this is to restrict the output phones to the desired phone subset
        if lang_id and lang_id != 'ipa':

            mask = self.inventory.get_mask(lang_id, approximation=self.config.approximate)

            logits = mask.mask_logits(logits)

            # print("masking ", str(mask_units))
        else:
            mask = self.inventory.unit

        emit_frame_idx = []
        cur_max_arg = -1

        # find all emitting frames
        for i in range(len(logits)):

            logit = logits[i]
            logit[0] /= blank_factor

            arg_max = np.argmax(logit)

            # this is an emitting frame
            if arg_max != cur_max_arg and arg_max != 0:
                emit_frame_idx.append(i)
                cur_max_arg = arg_max

        # decode all emitting frames
        decoded_seq = []
        for idx in emit_frame_idx:
            logit = logits[idx]
            top_phones = logit.argsort()[-topk:][::-1]
            decoded_seq.append(' '.join(mask.get_units(top_phones)))

        if topk == 1:
            phones = ' '.join(decoded_seq)
        else:
            phones = ' | '.join(decoded_seq)

        return phones
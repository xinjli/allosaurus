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

    def compute(self, logits, lang_id=None, blank_factor=1.0):
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

        # greedy decoding to find argmax phones
        decoded_seq = []

        for t in range(len(logits)):

            logit = logits[t]
            logit[0] /= blank_factor

            arg_max = np.argmax(logit)
            decoded_seq.append(arg_max)

        ids = [x[0] for x in groupby(decoded_seq)]

        # ignore when x is <blk> (0)
        cleaned_decoded_seq = [x for x in filter(lambda x: x != 0, ids)]

        phones = ' '.join(mask.get_units(cleaned_decoded_seq))
        return phones
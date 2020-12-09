from .articulatory import *
from .unit import *
from pathlib import Path

def read_prior(prior_path):

    prior = {}

    for i, line in open(str(prior_path), 'r', encoding='utf-8'):
        unit, prob = line.split()

        if i == 0:
            assert unit == '<blk>', 'first element should be blank'

        prior[unit] = np.log(prob)

    return prior



class UnitMask:

    def __init__(self, domain_unit, target_unit, approximation=False, inference_config=None):
        """
        MaskUnit provides interface to mask phones

        :param domain_unit: all available units (phones)
        :param target_unit: usually a subset of domain_unit
        """

        self.inference_config = inference_config

        self.domain_unit = domain_unit
        self.target_unit = target_unit

        # whether or not to use articulatory feature to map unseen units
        self.approximation = approximation

        # available index in all_unit
        self.valid_mask = set()

        # invalid index in all_unit
        self.invalid_mask = set()

        # index mapping from all_unit to target_unit
        self.unit_map = dict()

        # prior
        self.prior = np.zeros(len(self.domain_unit), dtype=np.float32)

        if self.inference_config and self.inference_config.prior and Path(self.inference_config.prior).exists():
            self.create_prior()

        # mask
        self.create_mask()

        if self.approximation:
            self.articulatory = Articulatory()
            self.approxmiate_phone()

        # create a mask for masking numpy array
        self.invalid_index_mask = sorted(list(self.invalid_mask))


    def __str__(self):
        return '<UnitMask: valid phone: ' + str(len(self.valid_mask)) + ', invalid phone: '+ str(len(self.invalid_mask))+'>'

    def __repr__(self):
        return self.__str__()


    def create_mask(self):

        # invalidate all phones first
        self.invalid_mask = set(range(1, len(self.domain_unit)))

        # <blank> is valid phone
        self.valid_mask.add(0)

        # <blank> should be mapped to <blank>
        self.unit_map[0] = 0

        # register all valid phones
        for target_idx, target_phone in self.target_unit.id_to_unit.items():
            if target_phone in self.domain_unit:
                domain_idx = self.domain_unit.get_id(target_phone)

                # this domain_idx is available
                self.valid_mask.add(domain_idx)

                # remove the domain_idx from the invalid_mask
                self.invalid_mask -= { domain_idx }

                # register this domain idx -> target_idx
                self.unit_map[domain_idx] = target_idx

    def create_prior(self):

        for line in open(str(self.inference_config.prior), 'r', encoding='utf-8'):
            phone, logprob = line.strip().split()

            if phone in self.domain_unit:
                domain_idx = self.domain_unit.get_id(phone)
                self.prior[domain_idx] = logprob

    def approxmiate_phone(self):

        # register all valid phones
        for target_idx, target_phone in self.target_unit.id_to_unit.items():
            if target_phone not in self.domain_unit:

                # find the most similar phone from the invalid set
                max_domain_idx = -1
                max_domain_score = -10000

                for domain_idx in self.invalid_mask:

                    domain_phone = self.domain_unit.get_unit(domain_idx)

                    #print("domain ", domain_phone, " target ", target_phone)
                    score = self.articulatory.similarity(domain_phone, target_phone)

                    if score >= max_domain_score:
                        max_domain_score = score
                        max_domain_idx = domain_idx


                assert max_domain_idx not in self.valid_mask
                assert max_domain_idx != -1

                # map max_domain_idx to target_idx
                self.invalid_mask -= { max_domain_idx }
                self.valid_mask.add(max_domain_idx)

                #print("target phone", target_phone, ' mapped to idx ', max_domain_idx)

                self.unit_map[max_domain_idx] = target_idx


    def print_maps(self):

        for domain_id, target_id in self.unit_map.items():
            print(self.domain_unit.get_unit(domain_id)+' --> '+self.target_unit.get_unit(target_id))


    def mask_logits(self, logits):
        # mask inavailable logits

        # apply mask
        logits[:,self.invalid_index_mask] = -100000000.0

        # apply prior
        logits += self.prior

        return logits


    def get_units(self, ids):
        """
        get unit from ids

        :param ids: elem_id list
        :return: a list of unit
        """

        unit_lst = []

        for idx in ids:

            assert idx in self.unit_map

            target_idx = self.unit_map[idx]

            unit_lst.append(self.target_unit.get_unit(target_idx))

        return unit_lst
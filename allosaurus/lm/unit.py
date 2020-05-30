import numpy as np

def read_unit(unit_path):
    # load unit from units.txt
    # units.txt should start from index 1 (because ctc blank is taking the 0 index)

    unit_to_id = dict()

    unit_to_id['<blk>'] = 0

    idx = 0

    for line in open(str(unit_path), 'r', encoding='utf-8'):
        fields = line.strip().split()

        assert len(fields) < 3

        if len(fields) == 1:
            unit = fields[0]
            idx += 1
        else:
            unit = fields[0]
            idx = int(fields[1])

        unit_to_id[unit] = idx

    unit = Unit(unit_to_id)
    return unit


class Unit:

    def __init__(self, unit_to_id):
        """
        Unit manages bidirectional mapping from unit to id and id to unit
        both are dict

        :param unit_to_id:
        """

        self.unit_to_id = unit_to_id
        self.id_to_unit = {}

        assert '<blk>' in self.unit_to_id and self.unit_to_id['<blk>'] == 0

        for unit, idx in self.unit_to_id.items():
            self.id_to_unit[idx] = unit

    def __str__(self):
        return '<Unit: ' + str(len(self.unit_to_id)) + ' elems>'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, idx):
        return self.id_to_unit[idx]

    def __len__(self):
        return len(self.id_to_unit)

    def __contains__(self, unit):
        if unit == ' ':
            unit = '<space>'

        return unit in self.unit_to_id

    def get_id(self, unit):

        # handle special units
        if unit == ' ':
            unit = '<space>'

        assert unit in self.unit_to_id, 'unit '+unit+'is not in '+str(self.unit_to_id)
        return self.unit_to_id[unit]

    def get_ids(self, units):
        """
        get index for a word list
        :param words:
        :return:
        """

        return [self.get_id(unit) for unit in units]

    def get_unit(self, id):
        assert id >= 0 and id in self.id_to_unit

        unit = self.id_to_unit[id]

        # handle special units
        if unit == '<space>':
            unit = ' '

        return unit

    def get_units(self, ids):
        """
        get unit from ids

        :param ids: elem_id list
        :return: a list of unit
        """

        return [self.get_unit(id) for id in ids]
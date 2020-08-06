from allosaurus.pm.kdict import read_matrix
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

class AllosaurusDataset(Dataset):

    def __init__(self, data_path):
        self.data_path = Path(data_path)

        required_files = ['feat.scp', 'token', 'feat.ark', 'shape']

        for required_file in required_files:
            assert (self.data_path / required_file).exists(), required_file+" does not exist, please run the preparation before fine-tuning"

        # read all tokens
        self.utt2token = {}
        self._read_token()

        # read all features and their shapes
        self.utt2offset = {}
        self.utt2shape = {}
        self.ark = None
        self._read_feat()

        # extract all valid utt_ids
        token_utt_set = set(self.utt2token.keys())
        feat_utt_set = set(self.utt2offset.keys())
        shape_utt_set = set(self.utt2shape.keys())
        self.utt_ids = list(set.intersection(token_utt_set, feat_utt_set, shape_utt_set))

        # sort all ids based on their their shape
        self.utt_ids.sort(key=lambda utt_id: self.utt2shape[utt_id][0], reverse=True)

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, item):

        utt_id = self.utt_ids[item]

        token = self.utt2token[utt_id]

        offset = self.utt2offset[utt_id]

        self.ark.seek(offset)

        feature = read_matrix(self.ark, np.float32)

        return (feature, token)

    def close(self):

        if self.ark:
            self.ark.close()

    def _read_token(self):
        """
        load token from file

        :param token_path:
        :return:
        """

        token_reader = open(str(self.data_path / 'token'), 'r', encoding='utf-8')

        self.utt2token = {}

        for line in token_reader:
            fields = line.strip().split()
            utt_id = fields[0]
            tokens = list(map(int, fields[1:]))

            # reject empty token or too long token
            if len(tokens) == 0 or len(tokens) > 1000:
                continue

            self.utt2token[utt_id] = tokens


    def _read_feat(self):
        """
        load offsets from feat.scp

        :return:
        """

        #####################################################################
        # read feature
        #####################################################################

        feat_reader = open(str(self.data_path / 'feat.scp'), 'r')
        self.utt2offset = {}
        for line in feat_reader:
            fields = line.strip().split()

            assert len(fields) == 2, " feat.scp should only contain two fields"
            utt_id = fields[0]

            feat = fields[1]
            p = feat.rfind(":")

            assert p >= 0, " offset pointer not found"
            offset = int(feat[p+1:])
            self.utt2offset[utt_id] = offset

        feat_reader.close()

        self.ark = open(self.data_path / 'feat.ark', 'rb')

        #####################################################################
        # read shape
        #####################################################################

        shape_reader = open(str(self.data_path / 'shape'), 'r')

        for line in shape_reader:
            fields = line.strip().split()
            utt_id = fields[0]

            shape = (int(fields[1]), int(fields[2]))

            self.utt2shape[utt_id] = shape

        shape_reader.close()
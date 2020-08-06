from allosaurus.am.dataset import AllosaurusDataset
import numpy as np

def read_loader(data_path, train_config):
    """
    create a dataloader for data_path

    :param data_path:
    :param train_config:
    :return:
    """

    return AllosaurusLoader(data_path, train_config)


class AllosaurusLoader:

    def __init__(self, data_path, train_config):

        self.train_config = train_config

        self.dataset = AllosaurusDataset(data_path)

        self.batch_lst = []

        self._prepare_batch()

    def __len__(self):
        return len(self.batch_lst)

    def close(self):
        self.dataset.close()

    def shuffle(self):
        np.random.shuffle(self.batch_lst)

    def read_batch(self, batch_idx):
        assert batch_idx < len(self.batch_lst), "batch_idx "+str(batch_idx)+" is too large!!"

        batch = self.batch_lst[batch_idx]

        return self._collate_batch(batch)

    def _collate_batch(self, batch):

        feat_lst = []
        token_lst = []

        for idx in batch:
            feat, token = self.dataset[idx]
            feat_lst.append(feat)
            token_lst.append(token)

        feat, feat_lengths = self._collate_feat(feat_lst)
        token, token_lengths = self._collate_token(token_lst)

        return (feat, feat_lengths), (token, token_lengths)

    def _collate_feat(self, feat_lst):

        batch_size = len(feat_lst)
        frame_size = feat_lst[0].shape[0]
        feat_size = feat_lst[0].shape[1]

        # collate feats
        feat_lengths = np.zeros(batch_size, dtype=np.int32)
        feat_tensor = np.zeros([batch_size, frame_size, feat_size], dtype=np.float32)
        for i, feat in enumerate(feat_lst):

            feat_tensor[i,:len(feat)] = feat
            feat_lengths[i] = len(feat)

        return feat_tensor, feat_lengths


    def _collate_token(self, token_lst):

        batch_size = len(token_lst)
        token_size = max([len(token) for token in token_lst])

        token_tensor = np.zeros([batch_size, token_size], dtype=np.int32)
        token_lengths = np.zeros(batch_size, dtype=np.int32)

        for i, token in enumerate(token_lst):
            token_tensor[i, :len(token)] = token
            token_lengths[i] = len(token)

        return token_tensor, token_lengths


    def _prepare_batch(self):

        batch = []
        batch_frame_size = 0

        for i in range(len(self.dataset)):
            utt_id = self.dataset.utt_ids[i]

            frame_size = self.dataset.utt2shape[utt_id][0]

            # batch frame is large enough
            if batch_frame_size + frame_size >= self.train_config.batch_frame_size:

                # commit current batch to the list
                self.batch_lst.append(batch)

                # reset frame size
                batch_frame_size = 0

                # reset batch
                batch = []

            batch_frame_size += frame_size
            batch.append(i)

        # commit the last batch if it is a valid batch
        if len(batch) > 0 and batch_frame_size < self.train_config.batch_frame_size:
            self.batch_lst.append(batch)
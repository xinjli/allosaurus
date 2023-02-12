from allosaurus.corpus import read_corpus, read_audio_corpus
from allosaurus.pm.model import read_pm
from allosaurus.lm.model import read_lm
import torch


def read_dataset(corpus_path, pm_config_or_name, lm_config_or_name=None, utt_cnt=None):

    pm = read_pm(pm_config_or_name)

    if lm_config_or_name is not None:
        corpus = read_corpus(corpus_path, utt_cnt=utt_cnt)
        lm = read_lm(lm_config_or_name)
    else:
        corpus = read_audio_corpus(corpus_path)
        lm = None

    return Dataset(corpus, pm, lm)


class Dataset:

    def __init__(self, corpus, pm, lm):
        self.corpus = corpus
        self.utt_ids = corpus.utt_ids

        self.pm = pm
        self.lm = lm

    def __repr__(self):
        if self.lm is not None:
            return f"<Dataset {self.corpus.corpus_id} utts: {len(self.corpus)} pm: {self.pm.model} lm: {self.lm.model} />"
        else:
            return f"<Audio Dataset {self.corpus.corpus_id} utts: {len(self.corpus)} pm: {self.pm.model} />"


    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, item_or_utt_id):

        if isinstance(item_or_utt_id, int):
            utt_id = self.utt_ids[item_or_utt_id]
        else:
            utt_id = item_or_utt_id

        # audio features
        audio = self.corpus.read_audio(utt_id, self.pm.config.sample_rate)
        feats = self.pm.compute(audio)

        # text features
        if self.lm is not None:
            text = self.corpus.read_text(utt_id)
            langs = self.lm.compute(text, self.corpus.lang_id)

            return {
                'utt_id': utt_id,
                'corpus_id': self.corpus.corpus_id,
                'lang_id': self.corpus.lang_id,
                'feats': feats,
                'langs': langs,
                'pm': self.pm.config,
                'lm': self.lm.config,
            }

        else:

            return {
                'utt_id': utt_id,
                'feats': feats,
                'pm': self.pm.config,
            }
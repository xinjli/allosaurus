from pathlib import Path
from allosaurus.config import allosaurus_config
from allosaurus.record import read_record
from allosaurus.text import read_text


def read_corpus_path(corpus_path_or_id):
    if not Path(corpus_path_or_id).exists():
        lang_id = corpus_path_or_id.split('_')[0]
        corpus_path = allosaurus_config.tmp_path / 'corpus' / lang_id / corpus_path_or_id
        return corpus_path
    else:
        return corpus_path_or_id



def read_corpus(corpus_path, utt_cnt=None):

    # corpus might be an corpus_id
    if not Path(corpus_path).exists():
        lang_id = corpus_path.split('_')[0]
        corpus_path = allosaurus_config.tmp_path / 'corpus' / lang_id / corpus_path

        assert corpus_path.exists(), corpus_path

    corpus_path = Path(corpus_path)
    record_path = corpus_path / 'record.txt'
    text_path = corpus_path / 'text.txt'

    # get corpus_id, lang_id
    corpus_id = corpus_path.name
    if '_' not in corpus_id:
        corpus_id = 'eng_unk'
        lang_id = 'eng'
    else:
        lang_id = corpus_id.split('_')[0]

    # load record if exists
    record = None
    if record_path.exists():
        record = read_record(record_path)

    # load text if exists
    text = None
    if text_path.exists():
        text = read_text(text_path, lang_id)

    assert (record is not None) or (text is not None), " both text and record are empty!"

    # extract utt_ids
    if record is None:
        utt_ids = text.utt_ids
    elif text is None:
        utt_ids = record.utt_ids
    else:
        utt_ids = sorted((set(text.utt_ids)).intersection(set(record.utt_ids)))
        if utt_cnt is not None:
            utt_ids = utt_ids[:utt_cnt]

    return Corpus(utt_ids, record, text, corpus_id, lang_id)


def read_audio_corpus(corpus_path, corpus_id='eng_unk', lang_id='eng', segment_duration=15.0):
    record = read_record(corpus_path, segment_duration=segment_duration)
    utt_ids = record.utt_ids
    text = None

    return Corpus(utt_ids, record, None, corpus_id, lang_id)


class Corpus:

    def __init__(self, utt_ids, record, text, corpus_id, lang_id):
        self.utt_ids = utt_ids
        self.record = record
        self.text = text

        self.corpus_id = corpus_id
        self.lang_id = lang_id

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<Corpus {self.corpus_id} >"

    def __len__(self):
        return len(self.utt_ids)

    def get_utt_id(self, key):
        if isinstance(key, int):
            assert key >=0 and key < len(self.utt_ids), str(key)+' is not a valid key'
            utt_id = self.utt_ids[key]
        else:
            utt_id = key

        return utt_id

    def __getitem__(self, key):
        utt_id = self.get_utt_id(key)
        return (self.record.read_audio(utt_id), self.text.read_text(utt_id))

    def read_audio(self, key, sample_rate=None):
        utt_id = self.get_utt_id(key)
        return self.record.read_audio(utt_id, sample_rate)

    def read_text(self, key):
        utt_id = self.get_utt_id(key)
        return self.text.read_text(utt_id)
from allosaurus.utils.jupyter import *


def read_text(text_path, lang_id='eng'):
    """
    read text

    :param text_path:
    :param kaldi: first column is utt_id if True
    :return:
    """

    utt_ids = []
    utt2text = dict()
    utt_cnt = 0

    fp = open(str(text_path), 'r', encoding='utf-8')
    lines = fp.readlines()

    for line in lines:
        fields = line.strip().split()

        utt_id = fields[0]
        sent = ' '.join(fields[1:])


        utt_ids.append(utt_id)
        utt2text[utt_id] = sent

        utt_cnt += 1


    utt_ids = sorted(utt_ids)

    text = Text(utt_ids, utt2text, lang_id)

    return text

def write_text(text, text_path):

    w = open(str(text_path), 'w', encoding='utf-8')

    for utt in sorted(text.utt_ids):
        sent = text[utt]
        w.write(utt+" "+sent+"\n")

    w.close()


def create_text(sents, lang_id):
    """
    create text object from sentences

    :param sents: list of str or list of list of str
    :return: text
    """


    # sents might be a one string
    if isinstance(sents, str):
        sents = [sents]

    utt_cnt = 0

    utt_ids = []
    utt2text = dict()

    for sent in sents:

        words = None

        if isinstance(sent, str):
            words = sent
        elif isinstance(sent, list):
            words = ' '.join(sent)
        else:
            raise NotImplementedError

        utt_id = f"utt_{utt_cnt:05d}"
        utt_ids.append(utt_id)
        utt2text[utt_id] = words

        utt_cnt += 1

    return Text(utt_ids, utt2text, lang_id)


def filter_text(text, utt_ids):

    new_utt_ids = []
    new_utt2text = dict()

    for utt_id in utt_ids:

        if utt_id in text.utt2text:
            new_utt_ids.append(utt_id)
            new_utt2text[utt_id] = text.utt2text[utt_id]

    return Text(new_utt_ids, new_utt2text, text.lang_id)



class Text:

    def __init__(self, utt_ids, utt2text, lang_id):
        """
        a Text instance manages the mapping from utt_id (str) to text (str)

        :param utt_ids:
        :param utt2text:
        """
        self.lang_id = lang_id

        self.utt_ids = utt_ids
        self.utt2text = utt2text

    def __str__(self):
        return "<Text: "+str(len(self.utt_ids))+" utterances>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key):

        if isinstance(key, int):
            assert key >=0 and key < len(self.utt_ids), str(key)+' is not a valid key'
            utt_id = self.utt_ids[key]
        else:
            utt_id = key
            assert utt_id in self.utt2text, "error: "+utt_id+" is not in text"

        return self.utt2text[utt_id]

    def _repr_html_(self):
        return dict2html(self.utt2text, ['utt_id', 'text'])

    def __len__(self):
        return len(self.utt2text)

    def __contains__(self, utt_id):
        return utt_id in self.utt2text

    def add(self, utt, words):
        self.utt2text[utt] = words

    def keys(self):
        return self.utt2text.keys()

    def values(self):
        return self.utt2text.values()

    def items(self):
        return self.utt2text.items()

    def read_text(self, utt_id: str) -> str:

        # utt2label has the key
        assert utt_id in self.utt2text, "error: "+utt_id+" is not in text"

        return self.utt2text[utt_id]

    def search(self, word):

        utt_ids = []

        for utt_id, words in self.utt2text.items():
            if word in words:
                utt_ids.append(utt_id)
                continue

        return utt_ids
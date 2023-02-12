import regex as re

class Cleaner:

    def __init__(self, config):

        self.config = config

        self.stopword_exp = re.compile(
                r'(http|ftp)|' # http or https
                r'(www)|'
                r'(\.com)|'
                r'(\.org)|'
                r'(\.edu)|'
                r'(\.lk)|'
                r'(\.rw)|'
                r'(t.co)|'
                r'(\[.*\])|'
                r'(\<.*\>)|'
                r'(\{.*\})|'
                r'localhost',
                re.IGNORECASE)

        self.emoji_exp = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   "]+", flags=re.UNICODE)

        self.number_exp = re.compile(r'\p{Number}+')

        #self.punc_exp  = re.compile(r'(\p{Punctuation})', re.IGNORECASE)

        self.alpha_exp = re.compile(r"[a-zA-Z]")
        self.other_exp = re.compile('[%s]' % re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~።፡()’'))

        self.word_breaker = None


    def ascii(self, utf_str):
        """
        filter utf-8 sentences and remove non ascii characters

        :param utf_str: string in utf-8
        :return: string in ascii
        """

        return utf_str.encode("ascii", errors="ignore").decode()


    def compute(self, sent: str, lang_id: str) -> str:

        # ASR cleaner
        for c in ("　", "「", "」", "『", "』", "・", "【", "】", "（", "）", "(", ")", "＝", "“", "„", "‘", "”", "；", "《", "》"):
            sent = sent.replace(c, "")

        sent = self.emoji_exp.sub(r'', sent)

        words = sent.split(' ')

        # remove nonstop words
        filtered_words = []
        for word in words:
            if not re.search(self.stopword_exp, word):
                filtered_words.append(word.lower())

        # remove punctuations
        filtered_sent = ' '.join(filtered_words)

        filtered_sent = self.other_exp.sub('', filtered_sent)

        return filtered_sent
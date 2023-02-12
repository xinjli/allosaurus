from phonepiece.inventory import *
import epitran


class Transliterator:

    def __init__(self, lang_id):
        """wrapper of epitran

        :param lang:
        :param backoff:
        """

        # language id
        self.lang_id = lang_id

        self.g2p_id = self.lang_id

        # use the default phoneme inventory
        self.phoneme = read_inventory(lang_id).phoneme

        self.dict = {}

        # special handling
        if lang_id == 'eng':

            r = open(PhonePieceConfig.data_path / 'lexicon/eng/lexicon.txt', 'r', encoding='utf-8')
            for line in r:
                fields = line.strip().split()
                word = fields[0]
                phones = fields[1:]
                self.dict[word] = phones

            self.epi = epitran.Epitran('eng-Latn')

        elif lang_id == 'cmn':
            # assume cmn-Hans
            print("cmn transliterator")
            cedict = PhonePieceConfig.data_path / 'lexicon' / 'cmn' / 'cedict_ts.u8'

            self.g2p_id = 'cmn-Hans'

            self.epi = epitran.Epitran('cmn-Hans', cedict_file=cedict)
        else:

            g2p_path = PhonePieceConfig.data_path / 'inventory' / lang_id / 'g2p'

            if g2p_path.exists():
                g2p_ids = sorted([p.stem for p in list(g2p_path.glob('*'))])
            else:
                g2p_ids = []

            assert len(g2p_ids) > 0, "provided lang_id "+lang_id+" is not supported by epitran"

            if len(g2p_ids) > 1:
                print(f"there are multiple g2p_ids {g2p_ids}, we use the first {g2p_ids[0]}")

            self.g2p_id = g2p_ids[0]

            self.epi = epitran.Epitran(self.g2p_id)
            print(f"language id: {self.g2p_id}")

        self.ignore_phonemes = ['̇', '-', "'"]


    def get_ipa(self, word, oov_writer=None):
        """Return IPA transliteration given by first acceptable mode.
        Args:
            token (unicode): orthographic text
        Returns:
            unicode: transliteration as Unicode IPA string
        """

        if word in self.dict:
            return self.dict[word]

        try:
            raw_ipa_list = self.epi.trans_list(word)
            ipa_list = []

            for phoneme in raw_ipa_list:
                if phoneme not in self.phoneme:

                    if phoneme in self.ignore_phonemes:
                        continue

                    warning_log = "WARNING: Transliterator: " + self.g2p_id + " not found phoneme " + phoneme + " in word " + word + " ipa_lst " + ' '.join(raw_ipa_list)
                    #print(warning_log)

                    nearest_phoneme = self.phoneme.get_nearest_phoneme(phoneme)

                    replace_log = "WARNING: use " + nearest_phoneme + " instead of " + phoneme
                    #print(replace_log)

                    if oov_writer:
                        oov_writer.write(warning_log+'\n')
                        oov_writer.write(replace_log+'\n')

                    ipa_list.append(nearest_phoneme)

                else:
                    ipa_list.append(phoneme)

        except:
            ipa_list = []
            print("WARNING: Transliterator: ", self.g2p_id, "not parse token ", word)

        self.dict[word] = ipa_list

        return ipa_list

    def get_id(self, word):
        ipa_lst = self.get_ipa(word)

        id_lst = self.phoneme.get_ids(ipa_lst)
        return id_lst
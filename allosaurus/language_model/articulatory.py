import panphon
import numpy as np

class Articulatory:

    def __init__(self):

        self.feature_table = panphon.FeatureTable()

    def feature(self, phone):

        try:
            feats = self.feature_table.word_to_vector_list(phone, numeric=True)
        except:
            if len(phone) == 2:
                phone = phone[0]+' '+phone[1]
                feats = self.feature_table.word_to_vector_list(phone, numeric=True)
            else:
                feats = []

        if len(feats) == 0:
            feats = np.zeros(24)
        else:
            feats = np.array(feats[0], dtype=np.float32)

        return feats

    def similarity(self, p1, p2):
        """
        similarity between phone 1 and phone 2

        :param p1:
        :param p2:
        :return:
        """
        return np.inner(self.feature(p1), self.feature(p2))

    def most_similar(self, target_phone, phone_cands):

        max_phone = None
        max_score = -1000000

        target_feature = self.feature(target_phone)

        for phone in phone_cands:
           score = np.inner(self.feature(phone), target_feature)

           if score > max_score:
               max_phone = phone
               max_score = score

        return max_phone
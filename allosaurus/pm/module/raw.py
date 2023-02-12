import torch


class Raw:

    def __init__(self, config):
        self.config = config
        self.model = self.config.model

    def __repr__(self):
        return "<raw preprocess>"

    def __str__(self):
        return self.__repr__()

    def compute(self, audio):
        return audio.samples
from allosaurus.audio import resample_audio
import torchaudio
import torch


class MFCC:

    def __init__(self, config):

        self.model = config.model

        # feature arch config
        self.config = config

        # sample rate
        self.sample_rate = config.sample_rate
        self.feature_size = config.feature_size

        self.transform = torchaudio.transforms.MFCC(sample_rate=self.sample_rate,
                                                    n_mfcc=self.feature_size)

    def __str__(self):
        return "MFCC ("+str(vars(self.config))+")"

    def __repr__(self):
        return self.__str__()

    def compute(self, audio):
        """
        return feature for audio

        :param audio:
        :return: mfcc feature
        """

        # do the resampling if necessary
        audio = resample_audio(audio, self.sample_rate)

        feat = self.transform(audio.samples).transpose(0,1)
        std, mean = torch.std_mean(feat, dim=1, keepdim=True)

        return (feat - mean) / std
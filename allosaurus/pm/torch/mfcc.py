import torch
import torchaudio


class TorchMFCC:

    def __init__(self, config):

        self.model = config.model

        # feature model config
        self.config = config

        # sample rate
        self.sample_rate = config.sample_rate

        # samples in each window
        self.window_size = int(config.window_size * config.sample_rate)

        # overlap between windows
        self.window_shift = int(config.window_shift * config.sample_rate)

        # feature window
        self.feature_window = config.feature_window

        # last complete window starting sample (index of sample)
        self.prev_window_sample = 0

        # last complete mfcc window index (index of window)
        self.prev_window_index = 0

        # list of mfcc features
        self.mfcc_windows = []

        # float32 or float64
        self.dtype = config.dtype

        self.original_sample_rate = self.sample_rate

        self.resampler = torchaudio.transforms.Resample(self.sample_rate, self.sample_rate)


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
        if self.sample_rate != audio.sample_rate:

            # default original is not correct, recreate Resample instance
            if self.original_sample_rate != audio.sample_rate:
                self.resampler = torchaudio.transforms.Resample(audio.sample_rate, self.sample_rate)
                self.original_sample_rate = audio.sample_rate

            samples = self.resampler(audio.samples)
        else:
            samples = audio.samples

        # get feature and convert into correct type (usually float32)
        feat = torchaudio.compliance.kaldi.fbank(
                samples, num_mel_bins=self.config.bank_size,
                sample_frequency=self.sample_rate, dither=0.0
        )

        # cmvn
        std, mean = torch.std_mean(feat)
        feat = (feat - mean) / std

        return feat
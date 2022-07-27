import wave
import numpy as np
from pathlib import Path
import resampy


def read_audio(filename, header_only=False, channel=0):
    """
    read_audio will read a raw wav and return an Audio object

    :param header_only: only load header without samples
    """

    if isinstance(filename, Path):
        filename = str(filename)

    wf = wave.open(filename)

    # initialize audio
    audio = Audio()

    # set stream basic info
    channel_number = wf.getnchannels()

    # check the input channel is valid
    assert channel < channel_number

    # set wav header
    audio.set_header(sample_rate=wf.getframerate(), sample_size=wf.getnframes(), channel_number=1,
                     sample_width=wf.getsampwidth())

    # set audio
    if not header_only:
        x = wf.readframes(wf.getnframes())

        assert (channel_number <= 2)

        audio_bytes = np.frombuffer(x, dtype='int16')

        # get the first channel if stereo
        if channel_number == 2:
            audio_bytes = audio_bytes[channel::2]

        audio.samples = audio_bytes

        # when some utils piping to stdout, sample size might not be correct (e.g: lame --decode)
        audio.sample_size = len(audio.samples)

    wf.close()

    return audio


def resample_audio(audio, target_sample_rate):
    """
    resample the audio by the target_sample_rate

    :param audio:
    :param target_sample_rate:
    :return:
    """

    # return the origin audio if sample rate is identical
    if audio.sample_rate == target_sample_rate:
        return audio

    new_samples = resampy.resample(audio.samples, audio.sample_rate, target_sample_rate)

    new_audio = Audio(new_samples, target_sample_rate)

    return new_audio


class Audio:

    def __init__(self, samples=None, sample_rate=8000):
        """
        Audio is the basic data structure used in this package.
        It is used to capture fundamental info about audio files such as frequency and samples.

        :param samples:
        :param sample_rate:
        :param stream_name:
        """

        # default parameters
        if samples is None:
            samples = []

        self.sample_rate = sample_rate
        self.channel_number = 1
        self.sample_width = 2

        # segments
        self.segments = []

        # all samples
        self.set_samples(samples)

    def __str__(self):
        wave_info = "<Audio sample rate: "+str(self.sample_rate)+", samples: "\
                    + str(self.sample_size) + ", second: " + str(self.sample_size/self.sample_rate) + " > "
        return wave_info

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.sample_size

    def set_samples(self, samples):
        self.samples = samples
        self.sample_size = len(samples)

    def empty(self):
        return self.samples is None or self.sample_size == 0

    def clear(self):
        self.set_samples([])

    def extend(self, new_audio):
        """
        extend wave stream

        :param new_audio:
        :return:
        """

        # resample if sample_rate does not match
        if self.sample_rate != new_audio.sample_rate:
            audio =  resample_audio(new_audio, self.sample_rate)
            samples = audio.samples

        else:
            samples = new_audio.samples

        # extend
        new_samples = np.append(self.samples, samples)
        self.set_samples(new_samples)


    def set_header(self, sample_rate=8000, sample_size=0, channel_number=1, sample_width=2):
        self.sample_rate = sample_rate
        self.sample_size = sample_size
        self.channel_number = channel_number
        self.sample_width = sample_width

    def duration(self):
        return self.sample_size/self.sample_rate
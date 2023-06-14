import struct
import numpy as np
from pathlib import Path
import torchaudio
import torchaudio.functional as F
import torch
from scipy.signal import gaussian
from scipy.signal import spectrogram as scipy_spectrogram
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
import wave
import contextlib
import os


def read_audio(filename_or_audio, sample_rate=16000):

    utt_id = "audio"

    if isinstance(filename_or_audio, Audio):
        orig_sample_rate = filename_or_audio.sample_rate
        samples = filename_or_audio.samples
        utt_id = filename_or_audio.utt_id
    elif isinstance(filename_or_audio, str) or isinstance(filename_or_audio, Path):
        samples, orig_sample_rate = torchaudio.load(filename_or_audio)
        utt_id = Path(filename_or_audio).stem

        # keep the first channel
        samples = samples[0]
    else:
        assert sample_rate is not None
        samples = torch.tensor(filename_or_audio)
        samples = samples
        orig_sample_rate = sample_rate

    if sample_rate is None:
        sample_rate = orig_sample_rate
    elif sample_rate != orig_sample_rate:
        samples = F.resample(samples, orig_sample_rate, sample_rate)

    return Audio(samples, sample_rate, utt_id=utt_id)


def read_audio_duration(filename):

    if str(filename).endswith('.wav'):
        with contextlib.closing(wave.open(str(filename), 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

    else:
        meta = torchaudio.info(filename)
        duration = meta.num_frames / meta.sample_rate

    return duration

def serialize_audio(audio):
    """serialize wav into bytes

    :param audio:
    :return:
    """

    data = struct.pack("h"*len(audio.samples), *list(np.int16(audio.samples*32768)))
    return data


def deserialize_audio(audio_bytes):
    samples = np.frombuffer(audio_bytes, dtype='int16')

    return samples


def is_audio_file(audio_path):

    if not os.path.isfile(audio_path):
        return False

    suffix = str(audio_path).split('.')[-1]
    return suffix in ["wav", "mp3", "flac", "sph", "ogg"]


def find_audio(audio_path, suffix=None):
    """
    utilities to find all audio files

    :param audio_path:
    :param suffix:
    :return:
    """

    if suffix is None:
        suffix = ["wav", "mp3", "flac", "sph", "ogg"]

    if isinstance(suffix, str):
        suffix = [suffix]

    audio_lst = []

    for ext in suffix:
        audio_lst.extend(list(Path(audio_path).glob("**/*."+ext)))

    audio_lst.sort()

    return audio_lst


def write_audio(audio, filename):
    """
    write out a audio object to a wav file

    :param audio: an instance of Audio
    :param filename: filename of wav
    :return:
    """

    # augment the channel dim
    samples = audio.samples.unsqueeze(0)

    torchaudio.save(str(filename), samples, sample_rate=audio.sample_rate, bits_per_sample=16, encoding='PCM_S')



def random_audio(sample_size, sample_rate):
    """
    create a random audio

    :param sample_size:  number of sample
    :param sample_rate: sample frequency
    :return:
    """

    samples = torch.from_numpy(np.random.randint(-32767, 32767, sample_size) / 32767)
    audio = Audio(samples, sample_rate)
    return audio


def silent_audio(second, sample_rate=16000):

    samples = torch.zeros(second*sample_rate)
    audio = Audio(samples, sample_rate)
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

    samples = audio.samples
    new_samples = F.resample(samples, audio.sample_rate, target_sample_rate)
    new_audio = Audio(new_samples, target_sample_rate)

    return new_audio


def split_audio(audio, duration, step=-1, second=True):
    """
    split an audio into multiple small audios with length of duration

    :param audio: input wave stream
    :param sample_start: start sample
    :param sample_end: end sample
    :param sample_size: how many samples in each small wave stream
    :param sample_step: how many samples to move forward for each iteration
    :return:
    """

    audio_lst = []
    sample_rate = audio.sample_rate

    if step == -1:
        step = duration

    if second:
        duration = int(duration * sample_rate)
        step = int(step * sample_rate)

    sample_start = 0
    sample_end = len(audio.samples)

    idx = 0

    while sample_start <= sample_end:

        new_audio = Audio()
        new_audio.set_header(sample_rate=audio.sample_rate, channel_number=audio.channel_number,
                             sample_width=audio.sample_width, sample_size=audio.sample_size)

        new_audio_samples = audio.samples[sample_start:sample_start+duration].clone()
        new_audio.set_samples(new_audio_samples)

        if audio.utt_id is not None and len(audio.utt_id) > 0:
            new_audio.utt_id = f"{audio.utt_id}#{idx:04d}"
        else:
            new_audio.utt_id = f"{idx:04d}"

        idx += 1
        audio_lst.append(new_audio)

        # inc sample start for next iteration
        sample_start += step

    return audio_lst


def split_stream_audio(audio, sample_step):
    """
    split an audio into multiple non-overlapping stream audios
    each stream audio starts from the first sample

    for example: splitting [1 2 3 4 5 6] with sample_step 2 will yield following audios
    1 2
    1 2 3 4
    1 2 3 4 5 6

    :param audio: target audio
    :param sample_step: how many sample to skip for each stream audio
    :return: stream audio list
    """

    audio_lst = []
    sample_start = 0
    sample_len = len(audio.samples)

    while sample_start < sample_len:

        new_audio = Audio()
        new_audio.set_header(sample_rate=audio.sample_rate, channel_number=audio.channel_number,
                             sample_width=audio.sample_width, sample_size=audio.sample_size)

        new_audio_samples = audio.samples[:sample_start+sample_step].copy()
        new_audio.set_samples(new_audio_samples)

        audio_lst.append(new_audio)

        # inc sample start for next iteration
        sample_start += sample_step

    return audio_lst


def slice_audio(audio, sample_start, sample_end, second=True):

    if second:
        sample_start = int(audio.sample_rate * sample_start)
        sample_end = int(audio.sample_rate * sample_end)

    new_samples = audio.samples[sample_start:sample_end].clone()
    new_audio = Audio(new_samples, audio.sample_rate)
    return new_audio


def concatenate_audio(audio_lst, sample_rate=None):

    print(audio_lst[0].sample_rate)

    if sample_rate is None:
        sample_rate = audio_lst[0].sample_rate

    new_audio = Audio(sample_rate=sample_rate)

    sample_lst = []

    for audio in audio_lst:

        # resample
        if audio.sample_rate != sample_rate:
            audio = resample_audio(audio, sample_rate)

        sample_lst.append(audio.samples)

    samples = torch.concat(sample_lst)
    new_audio.set_samples(samples)

    return new_audio


class Audio:

    def __init__(self, samples=None, sample_rate=8000, utt_id="audio"):
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

        # offset field indicating the offset sec from the audio start, mainly for the partial audio
        self.offset = 0.0

        self.utt_id = utt_id

    def __str__(self):
        wave_info = "<Audio sample rate: "+str(self.sample_rate)+", samples: "\
                    + str(self.sample_size) + ", second: " + str(self.sample_size/self.sample_rate) + " > "
        return wave_info

    def __repr__(self):
        return self.__str__()

    def _repr_html_(self):
        self.play()

    def __len__(self):
        return self.sample_size

    def set_samples(self, samples):
        self.samples = samples
        self.sample_size = len(samples)

    def empty(self):
        return self.samples is None or self.sample_size == 0 or len(self.samples) == 0

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
            audio = resample_audio(new_audio, self.sample_rate)
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
        return len(self.samples)/self.sample_rate

    def play(self, start=None, end=None, frame=False):

        import IPython.display as ipd

        if start is None:
            start = 0

        if end is None:
            end = len(self.samples)

        if frame:
            samples = self.samples[start:end]
        else:
            samples = self.samples[int(start*self.sample_rate):int(end*self.sample_rate)]

        ipd.display(ipd.Audio(samples, rate=self.sample_rate))


    def waveform(self, start=None, end=None, frame=False):

        import matplotlib.pyplot as plt

        if start is None:
            start = 0

        if end is None:
            end = len(self.samples)

        if frame:
            samples = self.samples[start:end]
        else:
            samples = self.samples[start*self.sample_rate:end*self.sample_rate]


        wav_data = samples / 32768

        n_samples = len(wav_data)
        total_duration = n_samples / self.sample_rate
        sample_times = np.linspace(0, total_duration, n_samples)

        # plot figures
        # plt.ylim(-0.035, 0.035)
        # plt.xlabel('time [s]' fontsize=12)
        # plt.ylabel('amplitude', fontsize=12)
        plt.rc('figure', figsize=(16, 4))
        plt.margins(x=0)
        plt.plot(sample_times, wav_data, color='k')

    def visualize(self, start=None, end=None, frame=False):

        import matplotlib.pyplot as plt

        if start is None:
            start = 0

        if end is None:
            end = len(self.samples)

        if frame:
            samples = self.samples[start:end]
        else:
            samples = self.samples[start*self.sample_rate:end*self.sample_rate]

        waveform = samples.numpy()

        num_frames = waveform.shape[0]
        time_axis = torch.arange(0, num_frames) / self.sample_rate

        figure, axes = plt.subplots(2, 1, figsize=(16, 8))

        wav_data = samples

        n_samples = len(wav_data)
        total_duration = n_samples / self.sample_rate
        sample_times = np.linspace(0, total_duration, n_samples)

        axes[0].margins(x=0)
        axes[0].plot(sample_times, wav_data, color='k')
        axes[0].xaxis.tick_top()
        plt.subplots_adjust(wspace=0, hspace=0)

        axes[1].specgram(waveform, Fs=self.sample_rate)
        plt.show(block=False)



    def spectrogram(self, start=None, end=None, frame=False, image_path=None):

        import matplotlib.pyplot as plt

        if start is None:
            start = 0

        if end is None:
            end = len(self.samples)

        if frame:
            samples = self.samples[start:end]
        else:
            samples = self.samples[start*self.sample_rate:end*self.sample_rate]

        window_dur = 0.005
        step_dur = window_dur / np.sqrt(np.pi) / 8.
        dyn_range = 120

        # convert window & step durations from seconds to numbers of samples (which is what
        # scipy.signal.spectrogram takes as input).
        window_nsamp = int(window_dur * self.sample_rate * 2)
        step_nsamp = int(step_dur * self.sample_rate)

        # make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
        # backward from window_nsamp we can calculate σ.
        window_sigma = (window_nsamp + 1) / 6
        window = gaussian(window_nsamp, window_sigma)

        # convert step size into number of overlapping samples in adjacent analysis samples
        noverlap = window_nsamp - step_nsamp

        # compute the power spectral density
        freqs, times, power = scipy_spectrogram(samples*32768, detrend=False, mode='psd', fs=self.sample_rate,
                                          scaling='density', noverlap=noverlap,
                                          window=window, nperseg=window_nsamp)

        p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air

        # set lower bound of colormap (vmin) from dynamic range. The upper bound defaults
        # to the largest value in the spectrogram, so we don't need to set it explicitly.
        dB_max = 10 * np.log10(power.max() / (p_ref ** 2))
        vmin = p_ref * 10 ** ((dB_max - dyn_range) / 10)

        # set default colormap, if none specified
        cmap = get_cmap('Greys')

        # create the figure if needed
        fig, ax = plt.subplots()

        # other arguments to the figure
        extent = (times.min(), times.max(), freqs.min(), freqs.max())

        # plot
        ax.imshow(power, origin='lower', aspect='auto',
                  norm=LogNorm(), extent=extent)

        ax.set_ylim(0, self.sample_rate // 2)
        ax.set_xlabel('time (s)')
        ax.set_ylabel('frequency (Hz)')

        if image_path is None:
            plt.rc('figure', figsize=(16, 4))
            plt.show()

        else:
            if isinstance(image_path, Path):
                image_path = str(image_path)

            plt.savefig(image_path, bbox_inches='tight')
            plt.gcf().clear()

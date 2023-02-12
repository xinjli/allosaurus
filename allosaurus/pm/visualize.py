import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from scipy.signal import spectrogram as scipy_spectrogram
from matplotlib.colors import LogNorm
from matplotlib.cm import get_cmap
from allosaurus.audio import *

plt.rc('figure', figsize=(16, 4))


def convert_wav_to_float(data):
    if data.dtype == np.uint8:
        data = (data - 128) / 128.
    elif data.dtype == np.int16:
        data = data / 32768.
    elif data.dtype == np.int32:
        data = data / 2147483648.
    return data




def waveform(audio_or_path, image_path=None, channel=1, start=None, end=None):
    plt.rc('figure', figsize=(16, 4))

    if isinstance(audio_or_path, str):
        audio = read_audio(audio_or_path, 8000, channel)
    else:
        audio = audio_or_path

    if start is not None and end is not None:
        audio = slice_audio(audio, start, end, second=True)


    wav_data = convert_wav_to_float(audio.samples)

    n_samples = len(wav_data)
    total_duration = n_samples / audio.sample_rate
    sample_times = np.linspace(0, total_duration, n_samples)

    # plot figures
    #plt.ylim(-0.035, 0.035)
    #plt.xlabel('time [s]' fontsize=12)
    #plt.ylabel('amplitude', fontsize=12)
    plt.plot(sample_times, wav_data, color='k')

    if image_path is None:
        plt.show()

    else:
        plt.savefig(str(image_path), bbox_inches='tight')
        plt.gcf().clear()


def spectrogram(audio_or_path, image_path=None, channel=1, start=None, end=None, window_dur=0.005, step_dur=None, dyn_range=120,
                         cmap=None, ax=None):

    if isinstance(audio_or_path, str):
        audio = read_audio(audio_or_path, 8000, channel)
    else:
        audio = audio_or_path

    if start is not None and end is not None:
        audio = slice_audio(audio, start, end, second=True)

    # set default for step_dur, if unspecified. This value is optimal for Gaussian windows.
    if step_dur is None:
        step_dur = window_dur / np.sqrt(np.pi) / 8.

    # convert window & step durations from seconds to numbers of samples (which is what
    # scipy.signal.spectrogram takes as input).
    window_nsamp = int(window_dur * audio.sample_rate * 2)
    step_nsamp = int(step_dur * audio.sample_rate)

    # make the window. A Gaussian filter needs a minimum of 6σ - 1 samples, so working
    # backward from window_nsamp we can calculate σ.
    window_sigma = (window_nsamp + 1) / 6
    window = gaussian(window_nsamp, window_sigma)

    # convert step size into number of overlapping samples in adjacent analysis samples
    noverlap = window_nsamp - step_nsamp

    # compute the power spectral density
    freqs, times, power = scipy_spectrogram(audio.samples, detrend=False, mode='psd', fs=audio.sample_rate,
                                      scaling='density', noverlap=noverlap,
                                      window=window, nperseg=window_nsamp)

    p_ref = 2e-5  # 20 μPa, the standard reference pressure for sound in air

    # set lower bound of colormap (vmin) from dynamic range. The upper bound defaults
    # to the largest value in the spectrogram, so we don't need to set it explicitly.
    dB_max = 10 * np.log10(power.max() / (p_ref ** 2))
    vmin = p_ref * 10 ** ((dB_max - dyn_range) / 10)

    # set default colormap, if none specified
    if cmap is None:
        cmap = get_cmap('Greys')
    # or if cmap is a string, get the actual object
    elif isinstance(cmap, str):
        cmap = get_cmap(cmap)

    # create the figure if needed
    if ax is None:
        fig, ax = plt.subplots()

    # other arguments to the figure
    extent = (times.min(), times.max(), freqs.min(), freqs.max())

    # plot
    ax.imshow(power, origin='lower', aspect='auto', cmap=cmap,
              norm=LogNorm(), extent=extent)

    ax.set_ylim(0, audio.sample_rate // 2)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('frequency (Hz)')

    if image_path is None:
        plt.show()

    else:
        if isinstance(image_path, Path):
            image_path = str(image_path)

        plt.savefig(image_path, bbox_inches='tight')
        plt.gcf().clear()

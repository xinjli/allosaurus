from pathlib import Path
import tqdm
from allosaurus.audio import is_audio_file, read_audio, find_audio, split_audio, read_audio_duration, slice_audio, Audio


def read_record(record_input, segment_duration=-1):
    """
    record_input can be one of the followings:

    - path to a wav.scp (or equivalently record.txt)
    - path to a audio dir
    - path to an audio file
    - a single audio object
    - list of audio objects
    - list of audio file paths
    """

    # when input is a single audio object or a single path to a audio file
    if isinstance(record_input, Audio) or is_audio_file(record_input):
        record_input = [record_input]

    # when input is a list of audio objects or a list of paths to audio files
    if isinstance(record_input, list):
        utt_ids = []
        utt2audio = {}

        for audio_or_file in record_input:
            if isinstance(audio_or_file, Audio):
                audio = audio_or_file
                utt_id = audio.utt_id
            else:
                audio_path = Path(audio_or_file)
                utt_id = audio_path.stem
                audio = read_audio(audio_path)

            if 0 < segment_duration <= audio.duration():
                audio_lst = split_audio(audio, duration=segment_duration)
                for sub_audio in audio_lst:
                    utt_id = sub_audio.utt_id
                    utt_ids.append(utt_id)
                    utt2audio[utt_id] = sub_audio

            else:
                utt_ids.append(utt_id)
                utt2audio[utt_id] = audio

        assert len(utt_ids) > 0, "no audio exists"
        return Record(utt_ids, utt2audio, segment_duration=segment_duration)


    record_path = Path(record_input)

    # assume this is a directory containing audio files
    if record_path.is_dir():
        audio_lst = find_audio(record_path)

        utt_ids = []
        utt2audio = {}

        for audiofile in tqdm.tqdm(audio_lst):
            utt_id = audiofile.stem
            register_and_segment_audio(utt_id, audiofile, utt_ids, utt2audio, segment_duration)

        return Record(utt_ids, utt2audio, segment_duration=segment_duration)


    # assume this is a path to a wav.scp
    utt_ids = []
    utt2audio = {}

    corpus_path = record_path.parent

    if (corpus_path / 'segments').exists():

        # load wav.scp first
        wav2audio = {}

        lines = open(record_path, 'r').readlines()

        # keep in memory if there is only one line
        keep_in_memory = False
        if len(lines) == 1:
            keep_in_memory = True

        for line in lines:
            fields = line.strip().split()
            wav_id = fields[0]
            audio = fields[1]

            if keep_in_memory:
                audio = read_audio(audio)

            wav2audio[wav_id] = audio

        # load segments
        for line in open(corpus_path / 'segments', 'r', encoding='utf-8'):
            fields = line.strip().split()
            utt_id = fields[0]
            wav_id = fields[1]
            audio = wav2audio[wav_id]
            start_time = float(fields[2])
            end_time = float(fields[3])

            if isinstance(audio, Audio):
                sub_audio = slice_audio(audio, start_time, end_time)
            else:
                sub_audio = f'{audio}%{"%07d" % int(start_time * 100)}-{"%07d" % int(end_time * 100)}'

            utt_ids.append(utt_id)
            utt2audio[utt_id] = sub_audio

        utt_ids = sorted(utt_ids)


    else:
        for line in tqdm.tqdm(open(record_path, 'r', encoding='utf-8').readlines()):
            fields = line.strip().split()

            utt_id = fields[0]
            audio_path = Path(fields[1])
            register_and_segment_audio(utt_id, audio_path, utt_ids, utt2audio, segment_duration)


    return Record(utt_ids, utt2audio, segment_duration)


def register_and_segment_audio(utt_id, audiofile, utt_ids, utt2audio, segment_duration):

    if segment_duration > 0:
        duration = read_audio_duration(audiofile)
        if duration > segment_duration:

            if duration % segment_duration < 0.05:
                num_segment = int(duration // segment_duration)
            else:
                num_segment = int(duration // segment_duration + 1)

            print(f"segmenting {audiofile} with duration {duration} into {num_segment} segments")
            for idx in range(num_segment):
                sub_utt_id = f"{utt_id}#{idx:04d}"
                sub_audiofile = f"{audiofile}#{idx:04d}"
                utt_ids.append(sub_utt_id)
                utt2audio[sub_utt_id] = sub_audiofile
        else:
            utt_ids.append(utt_id)
            utt2audio[utt_id] = audiofile

    else:
        utt_ids.append(utt_id)
        utt2audio[utt_id] = audiofile


class Record:

    def __init__(self, utt_ids, utt2audio, segment_duration=-1):
        self.utt_ids = utt_ids
        self.utt2audio = utt2audio
        self.segment_duration = segment_duration

    def __str__(self):
        return "<Record: "+str(len(self.utt_ids))+" utterances>"

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, key, sample_rate=None):

        if isinstance(key, int):
            assert key >=0 and key < len(self.utt_ids), str(key)+' is not a valid key'
            utt_id = self.utt_ids[key]
        else:
            utt_id = key
            assert utt_id in self.utt2audio, "error: "+utt_id+" is not in text"

        audio_or_audio_file = self.utt2audio[utt_id]

        if isinstance(audio_or_audio_file, Audio):
            audio = audio_or_audio_file
        else:
            segment_idx = -1

            audio_file = audio_or_audio_file

            if self.segment_duration > 0 and self.is_partial_path(audio_file):
                segment_idx = int(str(audio_file)[-4:])
                audio_file = str(audio_file)[:-5]

            if self.is_segment_path(audio_file):
                full_audio_file = audio_file[:-16]
                start_time = float(audio_file[-15:-8])/100
                end_time = float(audio_file[-7:])/100

            else:
                full_audio_file = audio_file
                start_time = None
                end_time = None

            audio = read_audio(full_audio_file, sample_rate)

            if segment_idx != -1:
                sample_start = segment_idx * self.segment_duration
                sample_end = (segment_idx+1) * self.segment_duration
                audio = slice_audio(audio, sample_start, sample_end, second=True)

            if start_time is not None:
                sample_start = start_time
                sample_end = end_time
                audio = slice_audio(audio, sample_start, sample_end, second=True)

        return audio

    def __len__(self):
        return len(self.utt2audio)

    def __contains__(self, utt_id):
        return utt_id in self.utt2audio

    def is_partial_path(self, audio_path):
        audio_path = str(audio_path)
        if len(audio_path) >= 5 and audio_path[-5]=='#' and str.isdigit(audio_path[-4:]):
            return True
        else:
            return False

    def is_segment_path(self, audio_path):
        audio_path = str(audio_path)
        if len(audio_path) > 16 and audio_path[-16] == '%' and audio_path[-8] == '-':
            return True
        else:
            return False

    def read_audio(self, utt_id, sample_rate=None):

        return self.__getitem__(utt_id, sample_rate)
from pathlib import Path
import tarfile
import tqdm
from allosaurus.audio import read_audio, find_audio, split_audio, read_audio_duration, slice_audio, Audio


def read_record(record_input, segment_duration=-1):
    """
    record_input can either of the followings:

    - path to a record.txt
    - path to a audio dir
    - path to an audio file
    - a single audio object
    - list of audio objects
    - list of audio file paths
    """

    if isinstance(record_input, Audio):
        record_input = [record_input]

    if isinstance(record_input, list):
        is_audio_list = True
        is_file_list = True

        for file in record_input:
            if not isinstance(file, Audio):
                is_audio_list = False
            if not isinstance(file, str) and not isinstance(file, Path):
                is_file_list = False

        assert is_audio_list or is_file_list, "record input should contains either Audio objects or file paths when given by list"

        utt_ids = []
        utt2audio = {}

        if is_audio_list:
            for audio in record_input:

                if 0 < segment_duration <= audio.duration():
                    audio_lst = split_audio(audio, duration=segment_duration)
                    for sub_audio in audio_lst:
                        utt_id = sub_audio.utt_id
                        utt_ids.append(utt_id)
                        utt2audio[utt_id] = sub_audio

                else:
                    utt_id = audio.utt_id
                    utt_ids.append(utt_id)
                    utt2audio[utt_id] = audio

        if is_file_list:
            for audio_file in record_input:
                audio_path = Path(audio_file)
                utt_id = audio_path.stem

                audio = read_audio(audio_file)

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

        return Record(utt_ids, utt2audio, None)

    record_path = record_input
    record_path = Path(record_path)

    # this is a single audio file
    if record_path.suffix in ['.wav', '.mp3', '.flac', '.ogg']:
        audio = read_audio(record_path)
        utt_ids = [audio.utt_id]
        utt2audio = {audio.utt_id: audio}
        tar_file = None
        return Record(utt_ids, utt2audio, tar_file)


    # assume this is a directory containing audio files
    if not str(record_path).endswith('record.txt') and not str(record_path).endswith('wav.scp') and record_path.is_dir() and not (record_path / 'record.txt').exists() and not (record_path / 'wav.scp').exists():
        audio_lst = find_audio(record_path)

        utt_ids = []
        utt2audio = {}

        for audiofile in tqdm.tdqm(audio_lst):
            utt_id = audiofile.stem
            register_and_segment_audio(utt_id, audiofile, utt_ids, utt2audio, segment_duration)


        tar_file = None
        return Record(utt_ids, utt2audio, tar_file)

    if record_path.is_dir():
        record_path = record_path / 'record.txt'

    assert record_path.exists(), " record.txt does not exist!!"

    audio_dir = record_path.parent / 'audio'
    audio_tar = record_path.parent / 'audio.tar'

    in_memory_audio = {}

    if audio_dir.exists():

        utt_ids = []
        utt2audio = {}

        for line in open(record_path, 'r', encoding='utf-8'):
            fields = line.strip().split()

            utt_id = fields[0]
            audio_path = audio_dir / fields[1]

            assert audio_path.exists(), " audio file "+str(audio_path)+" does not exist!"
            register_and_segment_audio(utt_id, audio_path, utt_ids, utt2audio, segment_duration)

        utt_ids = sorted(utt_ids)
        tar_file = None

    elif audio_tar.exists():

        utt_ids = []
        utt2audio = {}

        for line in open(record_path, 'r', encoding='utf-8'):
            fields = line.strip().split()

            utt_id = fields[0]
            audio_path = fields[1]
            register_and_segment_audio(utt_id, audio_path, utt_ids, utt2audio, segment_duration)

        utt_ids = sorted(utt_ids)
        tar_file = tarfile.open(audio_tar, 'r')

    else:
        utt_ids = []
        utt2audio = {}

        if str(record_path).endswith('record.txt'):
            for line in tqdm.tqdm(open(record_path, 'r', encoding='utf-8').readlines()):
                fields = line.strip().split()

                utt_id = fields[0]
                audio_path = Path(fields[1])

                #assert audio_path.exists(), " audio file "+str(audio_path)+" does not exist!"
                register_and_segment_audio(utt_id, audio_path, utt_ids, utt2audio, segment_duration)

        else:
            assert str(record_path).endswith('segments')

            # load wav.scp first
            wav2audio = {}

            if (record_path.parent / 'wav.scp').exists():
                for line in open(record_path.parent / 'wav.scp', 'r'):
                    fields = line.strip().split()
                    wav2audio[fields[0]] = fields[1]

            for line in open(record_path, 'r', encoding='utf-8'):
                fields = line.strip().split()
                utt_id = fields[0]
                wav_id = fields[1]

                if wav_id in wav2audio:
                    audio = wav2audio[wav_id]
                else:
                    audio = str(record_path.parent / (wav_id+'.wav'))

                    if audio not in in_memory_audio:
                        # cache those audios
                        in_memory_audio[audio] = read_audio(audio)

                start_time = float(fields[2])
                end_time = float(fields[3])
                sub_audio = f'{audio}%{"%07d" % int(start_time * 100)}-{"%07d" % int(end_time * 100)}'
                utt_ids.append(utt_id)
                utt2audio[utt_id] = sub_audio

        utt_ids = sorted(utt_ids)
        tar_file = None

    return Record(utt_ids, utt2audio, tar_file, segment_duration, in_memory_audio)


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

    def __init__(self, utt_ids, utt2audio, tar_file=None, segment_duration=-1, in_memory_audio=None):
        self.utt_ids = utt_ids
        self.utt2audio = utt2audio
        self.tar_file = tar_file
        self.segment_duration = segment_duration

        if in_memory_audio is None:
            self.in_memory_audio = {}
        else:
            self.in_memory_audio = in_memory_audio

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

        audio_file = self.utt2audio[utt_id]
        segment_idx = -1

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

        if self.tar_file is not None:
            full_audio_file = "./"+full_audio_file
            full_audio_file = self.tar_file.extractfile(full_audio_file)

        # check in memory cache first
        if full_audio_file in self.in_memory_audio:
            audio = self.in_memory_audio[full_audio_file]
        else:
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
from pathlib import Path
import tarfile
from allosaurus.audio import read_audio, find_audio, Audio


def read_record(record_input):
    """
    record_input can either of the followings:

    - path to a record.txt
    - path to a audio dir
    - list of audio objects
    - list of audio file paths
    """

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
                utt_id = audio.utt_id
                utt_ids.append(utt_id)
                utt2audio[utt_id] = audio

        if is_file_list:
            for audio_file in record_input:
                audio_path = Path(audio_file)
                utt_id = audio_path.stem
                utt_ids.append(utt_id)
                utt2audio[utt_id] = read_audio(audio_file)

        assert len(utt_ids) > 0, "no audio exists"

        return Record(utt_ids, utt2audio, None)

    record_path = record_input
    record_path = Path(record_path)

    # assume this is a directory containing audio files
    if not str(record_path).endswith('record.txt') and record_path.is_dir():
        audio_lst = find_audio(record_path)

        utt_ids = []
        utt2audio = {}

        for audiofile in audio_lst:
            utt_id = audiofile.stem
            utt_ids.append(utt_id)
            utt2audio[utt_id] = audiofile

        tar_file = None
        return Record(utt_ids, utt2audio, tar_file)

    if record_path.is_dir():
        record_path = record_path / 'record.txt'

    assert record_path.exists(), " record.txt does not exist!!"

    audio_dir = record_path.parent / 'audio'
    audio_tar = record_path.parent / 'audio.tar'

    if audio_dir.exists():

        utt_ids = []
        utt2audio = {}

        for line in open(record_path, 'r', encoding='utf-8'):
            fields = line.strip().split()

            utt_id = fields[0]
            audio_path = audio_dir / fields[1]

            assert audio_path.exists(), " audio file "+str(audio_path)+" does not exist!"

            utt_ids.append(utt_id)
            utt2audio[utt_id] = audio_path

        utt_ids = sorted(utt_ids)
        tar_file = None

    elif audio_tar.exists():

        utt_ids = []
        utt2audio = {}

        for line in open(record_path, 'r', encoding='utf-8'):
            fields = line.strip().split()

            utt_id = fields[0]
            audio_path = fields[1]

            utt_ids.append(utt_id)
            utt2audio[utt_id] = audio_path

        utt_ids = sorted(utt_ids)
        tar_file = tarfile.open(audio_tar, 'r')

    else:
        utt_ids = []
        utt2audio = {}

        for line in open(record_path, 'r', encoding='utf-8'):
            fields = line.strip().split()

            utt_id = fields[0]
            audio_path = Path(fields[1])

            #assert audio_path.exists(), " audio file "+str(audio_path)+" does not exist!"

            utt_ids.append(utt_id)
            utt2audio[utt_id] = audio_path

        utt_ids = sorted(utt_ids)
        tar_file = None

    return Record(utt_ids, utt2audio, tar_file)


class Record:

    def __init__(self, utt_ids, utt2audio, tar_file=None):
        self.utt_ids = utt_ids
        self.utt2audio = utt2audio
        self.tar_file = tar_file

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

        if self.tar_file is not None:
            audio_file = "./"+audio_file
            audio_file = self.tar_file.extractfile(audio_file)

        return read_audio(audio_file, sample_rate)

    def __len__(self):
        return len(self.utt2audio)

    def __contains__(self, utt_id):
        return utt_id in self.utt2audio

    def read_audio(self, utt_id, sample_rate=None):

        return self.__getitem__(utt_id, sample_rate)
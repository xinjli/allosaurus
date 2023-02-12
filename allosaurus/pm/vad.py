import collections
import contextlib
import sys
import wave
import webrtcvad
from allosaurus.audio import *


class VadFrame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration, partial_audio):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration
        self.partial_audio = partial_audio


class VAD:

    def __init__(self, mode=1, frame_duration_ms=30, padding_duration_ms=300):
        """
        :param mode: aggressiveness
        """

        self.vad = webrtcvad.Vad(mode)
        self.frame_duration_ms = frame_duration_ms
        self.padding_duration_ms = padding_duration_ms

    def generate_frame(self, audio):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        n = int(audio.sample_rate * (self.frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / audio.sample_rate) / 2.0

        frame_lst = []

        audio_bytes = serialize_audio(audio)

        while offset + n < len(audio_bytes):
            partial_audio = slice_audio(audio, timestamp, timestamp+duration, second=True)
            frame_lst.append(VadFrame(audio_bytes[offset:offset + n], timestamp+partial_audio.offset, duration, partial_audio))
            timestamp += duration
            offset += n

        return frame_lst

    def compute(self, audio_or_path):
        """
        :param audio: target audio
        :return: timestamp_lst, each data is (start_timestamp, end_timestamp)
        """

        if not isinstance(audio_or_path, Audio):
            audio = read_audio(audio_or_path, sample_rate=16000)
        else:
            audio = audio_or_path

        # resample into 16000 if larger than it
        if audio.sample_rate >= 16000:
            audio = resample_audio(audio, 16000)
        else:
            audio = resample_audio(audio, 8000)

        num_padding_frames = int(self.padding_duration_ms / self.frame_duration_ms)
        ring_buffer = collections.deque(maxlen=num_padding_frames)

        voiced_frames = []

        frames = self.generate_frame(audio)

        # whether is in speech region
        triggered = False

        # results
        timestamp_lst = []
        audio_lst = []

        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, audio.sample_rate)

            # sys.stdout.write('1' if is_speech else '0')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    #sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    #sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                    triggered = False

                    start_timestamp = voiced_frames[0].timestamp
                    end_timestamp = start_timestamp + sum([frame.duration for frame in voiced_frames])
                    timestamp_lst.append((start_timestamp, end_timestamp))
                    audio_lst.append(concatenate_audio([frame.partial_audio for frame in voiced_frames]))

                    ring_buffer.clear()
                    voiced_frames = []

        # if triggered:
        #     sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))

        if voiced_frames:
            start_timestamp = voiced_frames[0].timestamp
            end_timestamp = start_timestamp + sum([frame.duration for frame in voiced_frames])
            timestamp_lst.append((start_timestamp, end_timestamp))
            audio_lst.append(concatenate_audio([frame.partial_audio for frame in voiced_frames]))

        return timestamp_lst, audio_lst

#!/usr/bin/env python

from allosaurus.utils.reporter import *
from allosaurus.pm.vad import *
from pathlib import Path
import argparse

def clean_by_vad(f, out_dir):

    vad = VAD()

    audio = read_audio(str(f), sample_rate=None)

    segments, _ = vad.compute(audio)

    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / str(f.name)

    if len(segments) == 0:
        reporter.warning(f"{f.name} is a silent audio, skip this file")

    elif len(segments) == 1:
        # only
        reporter.info(f"{f.name}: looks OK")
        start_timestamp, end_timestamp = segments[0]

        new_audio = slice_audio(audio, start_timestamp, end_timestamp, second=True)

        write_audio(new_audio, path)
    else:
        reporter.warning(f"{f.name} has more than two segments !!")

        audio_lst = []

        for i, (start_timestamp, end_timestamp) in enumerate(segments):
            new_audio = slice_audio(audio, start_timestamp, end_timestamp, second=True)
            audio_lst.append(new_audio)

        new_audio = concatenate_audio(audio_lst)

        write_audio(new_audio, path)

def clean_tail(f, out_dir):

    vad = VAD()

    audio = read_audio(str(f), sample_rate=None)

    segments, _ = vad.compute(audio)

    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / str(f.name)

    if len(segments) == 0:
        reporter.warning(f"{f.name} is a silent audio, skip this file")
        return

    _, end_timestamp = segments[-1]

    new_audio = slice_audio(audio, 0, end_timestamp, second=True)
    write_audio(new_audio, path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="split audio into chunks by vad")
    parser.add_argument("-i", "--input",   required=True,  help="input dir or wave")
    parser.add_argument("-f", "--format",  default='wav', help='audio format, wav, mp3, flac...')
    parser.add_argument("-o", "--output",  required=True,  help="output dir")
    parser.add_argument("-m", "--mode",    default=1,       help="output dir")
    parser.add_argument('-t', '--tail_only', required=False, default=False, help="tail only")

    args = parser.parse_args()

    p = Path(args.input)
    out_dir = Path(args.output)

    mode = int(args.mode)
    vad = VAD(mode)

    if p.is_dir():
        for f in p.glob('*.'+args.format):

            if args.tail_only:
                clean_tail(f, out_dir)
            else:
                clean_by_vad(f, out_dir)

    else:
        clean_by_vad(p, out_dir)

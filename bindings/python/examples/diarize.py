# Copyright 2024-2026 Qualcomm
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GenieX-Bridge Diarize Example - Speaker Diarization

This example demonstrates how to use the GenieX-Bridge Python binding to perform speaker diarization on an audio file.
"""

import argparse
import logging
import os

from geniex import setup_logging
from geniex.diarize import Diarize


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='GenieX-Bridge Diarize Example')
    parser.add_argument(
        '-m',
        '--model',
        default='Geniex/Pyannote-NPU',
        help='Model id or path',
    )
    parser.add_argument(
        '--audio',
        required=True,
        help='Path to the input audio file',
    )
    parser.add_argument(
        '--min-speakers',
        type=int,
        default=0,
        help='Minimum number of speakers (0 for auto-detect)',
    )
    parser.add_argument(
        '--max-speakers',
        type=int,
        default=0,
        help='Maximum number of speakers (0 for no limit)',
    )
    parser.add_argument('--plugin-id', default=None, help='Plugin ID to use')
    parser.add_argument('--device', default=None, help='Device to run on (e.g., cpu, gpu, 0)')
    args = parser.parse_args()

    audio_path = os.path.expanduser(args.audio)

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f'Audio file not found: {audio_path}')

    diarize = Diarize.from_(
        model=os.path.expanduser(args.model),
        plugin_id=args.plugin_id,
        device_id=args.device,
    )

    result = diarize.infer(
        audio_path=audio_path,
        min_speakers=args.min_speakers if args.min_speakers > 0 else None,
        max_speakers=args.max_speakers if args.max_speakers > 0 else None,
    )

    print(f'Number of speakers: {result.num_speakers}')
    print(f'Duration: {result.duration:.2f}s')
    print(f'Number of segments: {len(result.segments)}')
    print('\nSegments:')
    for segment in result.segments:
        print(f'[{segment.start_time:.2f}s - {segment.end_time:.2f}s] {segment.speaker_label}')


if __name__ == '__main__':
    main()

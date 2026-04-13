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
GenieX-Bridge TTS Example - Text to Speech

This example demonstrates how to use the GenieX-Bridge Python binding to synthesize speech from text.
"""

import argparse
import logging
import os

from geniex import setup_logging
from geniex.tts import TTS


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='GenieX-Bridge TTS Example')
    parser.add_argument(
        '-m',
        '--model',
        default='Geniex/Qwen3-0.6B-GGUF',
        help='Model id or path',
    )
    parser.add_argument(
        '--text',
        required=True,
        help='Text to synthesize',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output audio file path (default: auto-generated)',
    )
    parser.add_argument(
        '--voice',
        default=None,
        help='Voice identifier (if supported by model)',
    )
    parser.add_argument(
        '--speed',
        type=float,
        default=1.0,
        help='Speech speed multiplier (default: 1.0)',
    )
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Audio sample rate (default: 22050)',
    )
    parser.add_argument('--plugin-id', default=None, help='Plugin ID to use')
    parser.add_argument('--device', default=None, help='Device to run on (e.g., cpu, gpu, 0)')
    args = parser.parse_args()

    # Generate output filename if not specified
    if args.output:
        output_path = os.path.expanduser(args.output)
    else:
        import time

        output_path = f'tts_output_{int(time.time())}.wav'

    # Create TTS instance
    tts = TTS.from_(
        model=os.path.expanduser(args.model),
        plugin_id=args.plugin_id,
        device_id=args.device,
    )

    # Synthesize speech
    result = tts.synthesize(
        text=args.text,
        output_path=output_path,
        voice=args.voice,
        speed=args.speed,
        sample_rate=args.sample_rate,
    )

    print(f'Audio saved to: {result.audio_path}')
    print(f'Duration: {result.duration_seconds:.2f}s')
    print(f'Sample rate: {result.sample_rate} Hz')
    print(f'Channels: {result.channels}')
    print(f'Number of samples: {result.num_samples}')


if __name__ == '__main__':
    main()

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
GenieX-Bridge ASR Example - Speech to Text

This example demonstrates how to use the GenieX-Bridge Python binding to transcribe an audio file
or perform real-time streaming transcription from microphone input.

Usage:
    # Non-streaming: transcribe audio file
    python asr.py --model model_id --audio audio.wav

    # Streaming: real-time microphone transcription
    python asr.py --model model_id --stream
"""

import argparse
import logging
import os
import subprocess
import sys

import numpy as np

from geniex import setup_logging
from geniex.asr import ASR, ASRStreamConfig


def transcribe_audio(asr: ASR, audio_path: str, language: str, timestamps: str, beam_size: int):
    """
    Transcribe an audio file.

    Args:
        asr: ASR instance
        audio_path: Path to audio file
        language: Language code
        timestamps: Timestamp granularity
        beam_size: Beam size for decoding
    """
    logging.info(f'Transcribing audio file: {audio_path}')

    result = asr.transcribe(
        audio_path=audio_path,
        language=language,
        timestamps=timestamps,
        beam_size=beam_size,
    )

    print('\n' + '=' * 50)
    print('Transcription Result:')
    print('=' * 50)
    print(result.transcript)
    print('=' * 50)


def stream_microphone(asr: ASR, language: str, beam_size: int, timestamps: str):
    """
    Stream audio from microphone for real-time transcription using sox.

    Args:
        asr: ASR instance
        language: Language code
        beam_size: Beam size for decoding
        timestamps: Timestamp granularity
    """
    SAMPLE_RATE = 16000
    CHUNK_SIZE = 4096

    # Check if sox is available
    try:
        subprocess.run(['sox', '--version'], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print('Error: sox is not installed or not in PATH.')
        print('Install sox:')
        print('  Windows: choco install sox  (or download from https://sourceforge.net/projects/sox)')
        print('  Linux: sudo apt-get install sox')
        sys.exit(1)

    config = ASRStreamConfig(
        sample_rate=SAMPLE_RATE,
        chunk_duration=2.0,
        overlap_duration=1.0,
        max_queue_size=10,
        buffer_size=CHUNK_SIZE,
        timestamps=timestamps,
        beam_size=beam_size,
    )

    print('\n' + '=' * 50)
    print(f'Streaming Mode - Language: {language}')
    print('Recording... Press Ctrl+C to stop and show final result')
    print('=' * 50 + '\n')

    # Build sox command for microphone input
    if sys.platform == 'win32':
        sox_cmd = ['sox', '-t', 'waveaudio', '-c', '1', '-r', str(SAMPLE_RATE), '-d', '-t', 's16', '-']
    else:  # Linux
        sox_cmd = ['sox', '-d', '-t', 's16', '-r', str(SAMPLE_RATE), '-c', '1', '-']

    transcription_buffer = []
    chunks_processed = [0]

    def on_transcription(text: str):
        transcription_buffer.append(text)
        chunks_processed[0] += 1
        try:
            import shutil

            term_width = shutil.get_terminal_size().columns
        except Exception:
            term_width = 80

        display_text = text if len(text) <= term_width - 10 else '...' + text[-(term_width - 13) :]
        print(f'\r[{chunks_processed[0]}] {display_text}', end='', flush=True)

    try:
        sox_process = subprocess.Popen(sox_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if sox started successfully
        import time

        time.sleep(0.2)
        if sox_process.poll() is not None:
            _, stderr = sox_process.communicate()
            error_msg = stderr.decode('utf-8', errors='ignore')
            print(f'Error: sox failed to start\n{error_msg}')
            sys.exit(1)

        if sox_process.stdout is None:
            raise RuntimeError('Failed to open sox stdout pipe.')

        with asr.stream(language=language, config=config) as stream:
            stream.start(on_transcription=on_transcription)

            try:
                while True:
                    audio_bytes = sox_process.stdout.read(CHUNK_SIZE * 2)
                    if not audio_bytes:
                        break
                    audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    stream.push_audio(audio_array.tolist())
            except Exception as e:
                logging.error(f'Error reading audio: {e}')

            stream.stop(graceful=True)

    except KeyboardInterrupt:
        print('\n')
        logging.info('Recording stopped by user')
    except Exception as e:
        print(f'\nError: {e}')
        logging.exception('Streaming error')
    finally:
        try:
            sox_process.terminate()
            sox_process.wait(timeout=2)
        except Exception:
            try:
                sox_process.kill()
            except Exception:
                pass

    if transcription_buffer:
        print('\n' + '=' * 50)
        print('Final Transcription Result:')
        print('=' * 50)
        print(' '.join(transcription_buffer))
        print('=' * 50)


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(
        description='GenieX-Bridge ASR Example - Speech to Text',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python asr.py --model Geniex/parakeet-npu --audio audio.wav
  python asr.py --model Geniex/parakeet-npu --stream
  python asr.py --model Geniex/parakeet-npu --audio audio.wav --language zh
        """,
    )
    parser.add_argument('-m', '--model', default='Geniex/parakeet-npu', help='Model id or path')
    parser.add_argument('--audio', help='Path to audio file')
    parser.add_argument('--stream', action='store_true', help='Stream from microphone')
    parser.add_argument('--language', default='en', help='Language code (en, zh, etc.)')
    parser.add_argument('--beam-size', type=int, default=5, help='Beam size for decoding')
    parser.add_argument('--timestamps', default='segment', help='Timestamps: none|segment|word')
    parser.add_argument('--plugin-id', default=None, help='Plugin ID (cpu_gpu or npu)')
    parser.add_argument('--device', default=None, help='Device (cpu, gpu, 0, etc.)')

    args = parser.parse_args()

    if not args.stream and not args.audio:
        parser.print_help()
        print('\nError: Either --audio or --stream must be provided')
        sys.exit(1)

    asr = ASR.from_(
        model=os.path.expanduser(args.model),
        plugin_id=args.plugin_id,
        device_id=args.device,
    )

    if args.stream:
        stream_microphone(asr, args.language, args.beam_size, args.timestamps)
    else:
        audio_path = os.path.expanduser(args.audio)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f'Audio file not found: {audio_path}')
        transcribe_audio(asr, audio_path, args.language, args.timestamps, args.beam_size)


if __name__ == '__main__':
    main()

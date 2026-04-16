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

#!/usr/bin/env python3

"""
GenieX-Bridge VLM Example - Llama Model Testing

This example demonstrates how to use the GenieX-Bridge Python binding to work with vision-language models.
It includes basic model initialization, text generation, streaming, and chat template functionality.
"""

import argparse
import io
import logging
import os
import shlex
from pathlib import Path
from typing import List, Optional, Tuple

from geniex import (
    GenerationConfig,
    ModelConfig,
    VlmChatMessage,
    VlmContent,
    setup_logging,
)
from geniex.vlm import VLM


def parse_media_from_input(
    user_input: str,
) -> Tuple[str, Optional[List[str]], Optional[List[str]]]:
    tokens = shlex.split(user_input, posix=False)
    image_exts = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp'}
    audio_exts = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}

    image_paths, audio_paths, prompt_parts = [], [], []

    for token in tokens:
        path = Path(os.path.expanduser(token))
        if path.exists():
            ext = path.suffix.lower()
            if ext in image_exts:
                image_paths.append(str(path))
            elif ext in audio_exts:
                audio_paths.append(str(path))
            else:
                prompt_parts.append(token)
        else:
            prompt_parts.append(token)

    prompt = ' '.join(prompt_parts).strip()
    return prompt, image_paths or None, audio_paths or None


def main():
    setup_logging(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description='GenieX-Bridge VLM Example')
    parser.add_argument(
        '-m',
        '--model',
        default='Geniex/Qwen2.5-Omni-3B-GGUF',
        help='Path to the VLM model',
    )
    parser.add_argument('--max-tokens', type=int, default=128, help='Maximum tokens to generate')
    parser.add_argument('--system', default='You are a helpful assistant.', help='System message')
    parser.add_argument('--plugin-id', default=None, help='Plugin ID to use')
    args = parser.parse_args()

    instance: VLM = VLM.from_(
        model=os.path.expanduser(args.model),
        config=ModelConfig(),
        plugin_id=args.plugin_id,
    )

    conversation: List[VlmChatMessage] = [
        VlmChatMessage(
            role='system',
            contents=[VlmContent(type='text', text=args.system)],
        )
    ]
    strbuff = io.StringIO()

    print("Multi-round conversation started. Type '/quit' or '/exit' to end.")
    print('=' * 50)

    while True:
        user_input = input('\nUser: ').strip()
        if not user_input:
            print("Please provide an input or type '/quit' to exit.")
            continue

        if user_input.startswith('/'):
            cmds = user_input.split()
            if cmds[0] in {'/quit', '/exit', '/q'}:
                print('Goodbye!')
                break
            elif cmds[0] in {'/reset', '/r'}:
                instance.reset()
                print('Conversation reset')
                continue
            else:
                print('Unknown command. Available commands: /quit, /exit, /reset')
                continue

        prompt, images, audios = parse_media_from_input(user_input)

        contents = []
        if prompt:
            contents.append(VlmContent(type='text', text=prompt))
        if images:
            for image in images:
                contents.append(VlmContent(type='image', text=image))
        if audios:
            for audio in audios:
                contents.append(VlmContent(type='audio', text=audio))
        conversation.append(VlmChatMessage(role='user', contents=contents))

        prompt = instance.apply_chat_template(conversation)
        strbuff.truncate(0)
        strbuff.seek(0)

        print('Assistant: ', end='', flush=True)
        gen = instance.generate_stream(
            prompt,
            config=GenerationConfig(max_tokens=args.max_tokens, image_paths=images, audio_paths=audios),
        )
        result = None
        try:
            while True:
                token = next(gen)
                print(token, end='', flush=True)
                strbuff.write(token)
        except StopIteration as e:
            result = e.value

        if result and hasattr(result, 'profile_data') and result.profile_data:
            print(f'\n{result.profile_data}')

        conversation.append(
            VlmChatMessage(
                role='assistant',
                contents=[VlmContent(type='text', text=strbuff.getvalue())],
            )
        )


if __name__ == '__main__':
    main()

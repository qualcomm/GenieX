# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
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

"""geniex-py console-script entry point.

This module also serves as a reference for how downstream users consume
the ``geniex`` package: it imports only from the public ``geniex``
surface and never reaches into private ``_ffi`` internals.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time

import geniex
from geniex import (
    AutoModelForCausalLM,
    GeniexError,
    GeniexVLM,
    _progress,
    get_device_list,
    get_plugin_list,
    init,
    resolve_device_map,
)

_mm = geniex.model_manager

# When stdout/stderr is a pipe on Windows the default codec is cp1252 and
# model tokens outside Latin-1 (UTF-8 continuation bytes, emoji, replacement
# chars) crash the print. Force UTF-8 whenever the stream supports it so
# callers piping through `tee`, pytest capture, etc. get clean output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, 'reconfigure'):
        try:
            _stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:  # noqa: BLE001 — best-effort
            pass

# Matches bare media paths in the user prompt (absolute, relative, or
# Windows drive-qualified). Mirrors the regex used by the Go CLI so
# behaviour is consistent across bindings.
_MEDIA_PATH_RE = re.compile(r'(?:[a-zA-Z]:)?(?:\./|/|\\)[\S\\ ]+?\.(?i:jpg|jpeg|png|webp|mp3|wav)\b')
_IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp')
_AUDIO_EXTS = ('.mp3', '.wav')


def _parse_media(prompt: str) -> tuple[str, list[str], list[str]]:
    """Strip media paths from ``prompt`` and return ``(prompt, images, audios)``.

    Only paths pointing at existing files are consumed; anything else stays
    in the prompt so the user sees an explicit error from the model rather
    than silent truncation.
    """
    images: list[str] = []
    audios: list[str] = []
    cleaned = prompt
    for match in _MEDIA_PATH_RE.findall(prompt):
        if not os.path.isfile(match):
            print(f'warning: file not found: {match}', file=sys.stderr)
            continue
        ext = os.path.splitext(match)[1].lower()
        if ext in _AUDIO_EXTS:
            audios.append(match)
        elif ext in _IMAGE_EXTS:
            images.append(match)
        cleaned = cleaned.replace(f"'{match}'", '').replace(match, '')
    return cleaned.strip(), images, audios


def _ensure_downloaded(
    model: str,
    quant: str | None,
    *,
    hub: str = 'auto',
    display_name: str | None = None,
    chipset: str | None = None,
    local_path: str | None = None,
) -> _mm.ModelPaths | None:
    if os.path.exists(model):
        return None

    if hub == 'aihub' and not display_name:
        raise SystemExit('error: --display-name is required when --hub aihub')
    if hub in ('localfs', 'local') and not local_path:
        raise SystemExit('error: --local-path is required when --hub localfs')

    result: dict = {}
    printer = _progress.default_progress_printer()

    def _worker():
        try:
            if hub in ('aihub', 'localfs', 'local'):
                _mm.pull(
                    model,
                    quant=quant,
                    hub=hub,
                    local_path=local_path,
                    hf_token=os.environ.get('GENIEX_HFTOKEN'),
                    chipset=chipset,
                    display_name=display_name,
                    on_progress=printer,
                )
                key = f'{model}:{quant}' if quant else model
                result['paths'] = _mm.get_paths(key)
            else:
                result['paths'] = _mm.ensure_cached(
                    model,
                    quant=quant,
                    hub=hub,
                    hf_token=os.environ.get('GENIEX_HFTOKEN'),
                    on_progress=printer,
                )
        except BaseException as e:  # noqa: BLE001 — forward to main thread
            result['error'] = e

    # Run the blocking pull off the main thread so Ctrl-C is delivered promptly.
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(timeout=0.1)
    except KeyboardInterrupt:
        _progress.finish(printer)
        sys.stderr.write('(aborted — partial download preserved; rerun to resume)\n')
        sys.stderr.flush()
        os._exit(130)
    _progress.finish(printer)
    if 'error' in result:
        raise result['error']
    return result.get('paths')


_ANSI = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None
_DIM = '\x1b[2m' if _ANSI else ''
_CYAN = '\x1b[36m' if _ANSI else ''
_GREEN = '\x1b[32m' if _ANSI else ''
_RESET = '\x1b[0m' if _ANSI else ''


def _run_turn(
    model,
    history: list[dict],
    user: str,
    max_tokens: int,
    temperature: float,
    *,
    is_vlm: bool,
) -> None:
    images: list[str] = []
    audios: list[str] = []
    if is_vlm:
        user, images, audios = _parse_media(user)
        # VLM chat templates need typed content parts (one per image / audio
        # / text) so the tokenizer can emit the right marker tokens.
        parts: list[dict] = []
        for img in images:
            parts.append({'type': 'image', 'image': img})
        for aud in audios:
            parts.append({'type': 'audio', 'audio': aud})
        if user:
            parts.append({'type': 'text', 'text': user})
        history.append({'role': 'user', 'content': parts or user})
    else:
        history.append({'role': 'user', 'content': user})
    prompt = model.tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    reply_parts: list[str] = []
    gen_kwargs: dict = {
        'max_new_tokens': max_tokens,
        'temperature': temperature,
        'stream': True,
    }
    if is_vlm:
        gen_kwargs['images'] = images
        gen_kwargs['audios'] = audios
    streamer = model.generate(prompt, **gen_kwargs)
    try:
        for chunk in streamer:
            print(chunk, end='', flush=True)
            reply_parts.append(chunk)
        print()
    except KeyboardInterrupt:
        # Ctrl-C mid-generation: ask the C side to stop at the next token
        # boundary, then drain the queue so the streamer thread publishes
        # its final profile (stop_reason='cancelled').
        streamer.cancel()
        try:
            for chunk in streamer:
                print(chunk, end='', flush=True)
                reply_parts.append(chunk)
        except BaseException:
            pass
        print()

    if streamer.output is not None:
        p = streamer.output.profile
        stop = p.stop_reason or 'unknown'
        print(
            f'\n{_CYAN}— {p.decode_speed:.1f} tok/s · {p.generated_tokens} tok'
            f' · {p.ttft / 1e6:.1f} s first token · stop: {stop} —{_RESET}\n'
        )
    history.append({'role': 'assistant', 'content': ''.join(reply_parts)})


def _chat_loop(
    model,
    system: str | None,
    max_tokens: int,
    temperature: float,
    *,
    is_vlm: bool,
) -> None:
    history: list[dict] = []
    if system:
        history.append({'role': 'system', 'content': system})

    while True:
        # Ctrl-C at the prompt clears the line and re-prompts (matching
        # llama-cli / ollama run); only Ctrl-D ends the session.
        try:
            user = input(f'{_GREEN}> {_RESET}')
        except KeyboardInterrupt:
            print()
            continue
        except EOFError:
            print()
            return

        text = user.strip()
        if not text:
            continue
        if text in ('/exit', '/quit'):
            return
        if text == '/reset':
            history = [history[0]] if system else []
            model.reset()
            print(f'{_DIM}(history cleared){_RESET}')
            continue

        _run_turn(model, history, user, max_tokens, temperature, is_vlm=is_vlm)


def _cmd_devices(_args: argparse.Namespace) -> int:
    init()
    plugins = get_plugin_list()
    if not plugins:
        print('No plugins available.')
        return 0
    for plugin_id in plugins:
        devices = get_device_list(plugin_id)
        print(f'{plugin_id}:')
        if not devices:
            print('  (no devices)')
            continue
        for dev_id, dev_name in devices:
            print(f'  {dev_id:<16} {dev_name}')
    return 0


def _cmd_chat(args: argparse.Namespace) -> int:
    _ensure_downloaded(
        args.model,
        args.quant,
        hub=args.hub,
        display_name=args.display_name,
        chipset=args.chipset,
        local_path=args.local_path,
    )

    name = f'{args.model}:{args.quant}' if args.quant else args.model
    sys.stdout.write(f'{_DIM}loading {name} ...{_RESET} ')
    sys.stdout.flush()
    t0 = time.monotonic()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quant=args.quant,
        device_map=args.device,
        n_ctx=args.n_ctx,
    )
    elapsed = time.monotonic() - t0
    is_vlm = isinstance(model, GeniexVLM)
    model_type = 'vlm' if is_vlm else 'llm'
    plugin_id, device_id, _ngl = resolve_device_map(args.device, args.model)
    where = f'{plugin_id}:{device_id}' if plugin_id and device_id else (plugin_id or args.device)
    print(f'{_DIM}done ({model_type}, {elapsed:.1f}s, {where}){_RESET}')

    try:
        if args.prompt is not None:
            history: list[dict] = []
            if args.system:
                history.append({'role': 'system', 'content': args.system})
            _run_turn(model, history, args.prompt, args.max_tokens, args.temperature, is_vlm=is_vlm)
        else:
            if is_vlm:
                print(
                    f'{_DIM}VLM mode — drop image/audio paths into the prompt '
                    f'(png/jpg/webp/mp3/wav) to attach media.{_RESET}'
                )
            print(f'{_DIM}Use Ctrl+D or /exit to exit.{_RESET}\n')
            _chat_loop(model, args.system, args.max_tokens, args.temperature, is_vlm=is_vlm)
    finally:
        model.close()
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    _ensure_downloaded(
        args.model,
        args.quant,
        hub=args.hub,
        display_name=args.display_name,
        chipset=args.chipset,
        local_path=args.local_path,
    )
    return 0


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024:
            return f'{int(size)} {unit}' if unit == 'B' else f'{size:.0f} {unit}'
        size /= 1024
    return f'{size:.1f} PiB'


def _read_manifest(name: str) -> dict | None:
    try:
        paths = _mm.get_paths(name)
    except GeniexError:
        return None
    try:
        with open(os.path.join(paths.model_dir, 'geniex.json')) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _render_table(rows: list[list[str]], headers: list[str]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))
    border = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    def _row(cells: list[str]) -> str:
        return '| ' + ' | '.join(c.ljust(w) for c, w in zip(cells, widths)) + ' |'

    print(border)
    print(_row(headers))
    print(border)
    for r in rows:
        print(_row(r))
    print(border)


def _cmd_ls(args: argparse.Namespace) -> int:
    if args.model:
        manifest = _read_manifest(args.model)
        if manifest is None:
            print(f'error: {args.model} is not cached', file=sys.stderr)
            return 1
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    names = _mm.list_models()
    if not names:
        print(f'{_DIM}(no models cached){_RESET}')
        return 0

    def _file_size(f: dict) -> int:
        # Size may be -1 for LocalFs pulls where the hub couldn't stat ahead.
        s = int(f.get('Size') or 0)
        return max(s, 0)

    rows: list[list[str]] = []
    for name in names:
        manifest = _read_manifest(name) or {}
        try:
            paths = _mm.get_paths(name)
            model_dir = paths.model_dir
        except GeniexError:
            model_dir = None

        size = 0
        quants = sorted((manifest.get('ModelFile') or {}).keys())

        def _add(f: dict) -> None:
            nonlocal size
            if not f.get('Downloaded'):
                return
            s = _file_size(f)
            if s == 0 and model_dir and f.get('Name'):
                try:
                    s = os.path.getsize(os.path.join(model_dir, f['Name']))
                except OSError:
                    s = 0
            size += s

        for f in (manifest.get('ModelFile') or {}).values():
            _add(f)
        _add(manifest.get('MMProjFile') or {})
        _add(manifest.get('TokenizerFile') or {})
        for f in manifest.get('ExtraFiles') or []:
            _add(f)

        rows.append(
            [
                name,
                _human_size(size) if size else '?',
                manifest.get('PluginId', '') or '',
                manifest.get('ModelType', '') or '',
                ','.join(quants) or '-',
            ]
        )

    _render_table(rows, ['NAME', 'SIZE', 'PLUGIN', 'TYPE', 'QUANTS'])
    return 0


def _cmd_rm(args: argparse.Namespace) -> int:
    if args.all:
        n = _mm.clean()
        print(f'removed {n} model{"s" if n != 1 else ""}')
        return 0
    if not args.model:
        print('error: specify a model name or --all', file=sys.stderr)
        return 2
    _mm.remove(args.model)
    print(f'removed {args.model}')
    return 0


def _add_hub_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        '--hub',
        choices=['auto', 'hf', 'huggingface', 'aihub', 'localfs', 'local'],
        default='auto',
        help='Source hub (default: auto = HuggingFace)',
    )
    p.add_argument(
        '--display-name',
        default=None,
        help='AI Hub model display_name (required when --hub aihub)',
    )
    p.add_argument(
        '--chipset',
        default=None,
        help=('AI Hub target chipset (e.g. qualcomm-snapdragon-x-elite); omit to auto-detect on Windows-on-Snapdragon'),
    )
    p.add_argument(
        '--local-path',
        default=None,
        help='Source directory (required when --hub localfs)',
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='geniex-py', description='GenieX Python CLI')
    sub = parser.add_subparsers(dest='cmd', required=True)

    chat = sub.add_parser('chat', help='Interactive chat with a model')
    chat.add_argument('model', help='Alias (e.g. qwen3), HF repo id, or local path')
    chat.add_argument('--quant', default=None, help='Quantization variant (e.g. Q4_K_M)')
    chat.add_argument('--system', default=None, help='Optional system prompt')
    chat.add_argument(
        '-p',
        '--prompt',
        default=None,
        help='Run a single turn with this prompt and exit (non-interactive)',
    )
    chat.add_argument('--max-tokens', type=int, default=512)
    chat.add_argument('--temperature', type=float, default=0.7)
    chat.add_argument('--n-ctx', type=int, default=0, help='Context length (0 = model default)')
    chat.add_argument(
        '--device',
        default='auto',
        help=(
            "'auto' | 'cpu' | 'gpu' | 'npu' | 'hybrid' | '<plugin>' | "
            "'<plugin>:<device>' (default: hybrid for llama_cpp, npu for "
            "qairt; run 'geniex-py devices' to list concrete ids)"
        ),
    )
    _add_hub_args(chat)
    chat.set_defaults(func=_cmd_chat)

    pull = sub.add_parser('pull', help='Download a model into the local cache')
    pull.add_argument('model', help='Alias or HF repo id (supports org/repo:quant)')
    pull.add_argument('--quant', default=None, help='Quantization variant (e.g. Q4_K_M)')
    _add_hub_args(pull)
    pull.set_defaults(func=_cmd_pull)

    ls = sub.add_parser('ls', help='List cached models, or show one model as JSON')
    ls.add_argument('model', nargs='?', help='Show this model as JSON instead of a table')
    ls.set_defaults(func=_cmd_ls)

    rm = sub.add_parser('rm', help='Remove a cached model')
    rm.add_argument('model', nargs='?', help='Model name to remove')
    rm.add_argument('--all', action='store_true', help='Remove every cached model')
    rm.set_defaults(func=_cmd_rm)

    devices = sub.add_parser('devices', help='List available plugins and devices')
    devices.set_defaults(func=_cmd_devices)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except GeniexError as e:
        print(f'error: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

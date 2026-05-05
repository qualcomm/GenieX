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

"""Minimal ``geniex`` command-line entry point.

Installed as a console script via ``[project.scripts]`` in pyproject.toml;
the library-level API (``geniex.AutoModelForCausalLM`` etc.) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

from . import model_manager as _mm
from ._ffi._api import GeniexError, ensure_init, get_device_list, get_plugin_list
from .auto import AutoModelForCausalLM, _resolve_device


def _fmt_bytes(n: int) -> str:
    if n < 0:
        return '?'
    size = float(n)
    for unit in ('B', 'KiB', 'MiB', 'GiB'):
        if size < 1024:
            return f'{size:.1f}{unit}' if unit != 'B' else f'{int(size)}B'
        size /= 1024
    return f'{size:.1f}TiB'


def _make_progress_printer():
    """Return an on_progress callback that renders a single-line status to stderr."""

    def _cb(files):
        parts = []
        for f in files:
            total = f.total_bytes
            got = f.downloaded_bytes
            pct = f' {got * 100 // total}%' if total > 0 else ''
            parts.append(f'{f.file_name} {_fmt_bytes(got)}/{_fmt_bytes(total)}{pct}')
        line = ' | '.join(parts)
        # Pad so shorter lines overwrite longer previous lines cleanly.
        sys.stderr.write('\r' + line.ljust(80)[:120])
        sys.stderr.flush()
        return True

    return _cb


def _ensure_downloaded(model: str, quant: str | None) -> _mm.ModelPaths | None:
    """Download the model through the model manager if it's not a local path.

    Returns the resolved :class:`ModelPaths` so callers can hand a local
    file to ``from_pretrained`` and skip a second model-manager round-trip
    (which would otherwise hit ``repo.info()`` again over the network).
    Returns ``None`` when ``model`` is already a local path.

    The Rust pull is a blocking FFI call, so we run it in a daemon thread
    and keep the main thread in Python so Ctrl-C is delivered promptly.
    The ``.inflight/`` sentinel stays on disk if the user aborts, and a
    subsequent run will resume from where it left off.
    """
    if os.path.exists(model):
        return None

    result: dict = {}

    def _worker():
        try:
            result['paths'] = _mm.ensure_cached(
                model,
                quant=quant,
                hub='auto',
                hf_token=os.environ.get('GENIEX_HFTOKEN'),
                on_progress=_make_progress_printer(),
            )
        except BaseException as e:  # noqa: BLE001 — forward to main thread
            result['error'] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        while t.is_alive():
            t.join(timeout=0.1)
    except KeyboardInterrupt:
        sys.stderr.write('\n(aborted — partial download preserved; rerun to resume)\n')
        sys.stderr.flush()
        os._exit(130)
    sys.stderr.write('\n')
    sys.stderr.flush()
    if 'error' in result:
        raise result['error']
    return result.get('paths')


_ANSI = sys.stdout.isatty() and os.environ.get('NO_COLOR') is None
_DIM = '\x1b[2m' if _ANSI else ''
_CYAN = '\x1b[36m' if _ANSI else ''
_GREEN = '\x1b[32m' if _ANSI else ''
_RESET = '\x1b[0m' if _ANSI else ''


def _run_turn(model, history: list[dict], user: str, max_tokens: int, temperature: float) -> None:
    """Stream one assistant reply and append it to ``history`` in place."""
    history.append({'role': 'user', 'content': user})
    prompt = model.tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    reply_parts: list[str] = []
    streamer = model.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )
    try:
        for chunk in streamer:
            print(chunk, end='', flush=True)
            reply_parts.append(chunk)
        print()
    except KeyboardInterrupt:
        # Ctrl-C during generation: ask the C side to stop at the next
        # token boundary (the plugin's decode loop breaks when our token
        # callback returns False), then drain whatever's already queued
        # so the streamer thread finishes and publishes its final
        # profile (including stop_reason='cancelled').
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
            f' · {p.ttft / 1000:.1f} s first token · stop: {stop} —{_RESET}\n'
        )
    history.append({'role': 'assistant', 'content': ''.join(reply_parts)})


def _chat_loop(model, system: str | None, max_tokens: int, temperature: float) -> None:
    history: list[dict] = []
    if system:
        history.append({'role': 'system', 'content': system})

    while True:
        # Ctrl-C at the prompt clears the half-typed line and re-prompts,
        # matching llama-cli / ollama run. Only Ctrl-D (EOF) ends the session.
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

        _run_turn(model, history, user, max_tokens, temperature)


def _cmd_devices(_args: argparse.Namespace) -> int:
    """List plugins and the devices each plugin exposes on this host."""
    ensure_init()
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
    paths = _ensure_downloaded(args.model, args.quant)

    name = f'{args.model}:{args.quant}' if args.quant else args.model
    sys.stdout.write(f'{_DIM}loading {name} ...{_RESET} ')
    sys.stdout.flush()
    t0 = time.monotonic()
    # After a successful pull, hand the resolved file path directly to
    # from_pretrained so it hits the local-path fast-path instead of
    # re-invoking ensure_cached (which would repeat the hub list_files).
    load_target = paths.model_path if paths else args.model
    model = AutoModelForCausalLM.from_pretrained(
        load_target,
        quant=None if paths else args.quant,
        device_map=args.device,
        n_ctx=args.n_ctx,
    )
    elapsed = time.monotonic() - t0
    plugin_id, device_id, _ngl = _resolve_device(args.device)
    where = f'{plugin_id}:{device_id}' if plugin_id and device_id else (plugin_id or args.device)
    print(f'{_DIM}done ({elapsed:.1f}s, {where}){_RESET}')

    try:
        if args.prompt is not None:
            history: list[dict] = []
            if args.system:
                history.append({'role': 'system', 'content': args.system})
            _run_turn(model, history, args.prompt, args.max_tokens, args.temperature)
        else:
            print(f'{_DIM}Use Ctrl+D or /exit to exit.{_RESET}\n')
            _chat_loop(model, args.system, args.max_tokens, args.temperature)
    finally:
        model.close()
    return 0


def _cmd_pull(args: argparse.Namespace) -> int:
    """Download a model into the local cache."""
    _ensure_downloaded(args.model, args.quant)
    return 0


def _human_size(n: int) -> str:
    size = float(n)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if size < 1024:
            return f'{int(size)} {unit}' if unit == 'B' else f'{size:.0f} {unit}'
        size /= 1024
    return f'{size:.1f} PiB'


def _read_manifest(name: str) -> dict | None:
    """Load the on-disk geniex.json for a cached model (or None on failure)."""
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
    """List cached models (or show one model's details when a name is given)."""
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
        """Return the file's size, falling back to disk stat when the manifest
        recorded -1 (e.g. LocalFs pulls where the hub couldn't stat ahead)."""
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
    """Remove one cached model, or all with --all."""
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
            "'auto' | 'cpu' | 'gpu' | 'npu' | '<plugin>' | '<plugin>:<device>' "
            "(run 'geniex-py devices' to list concrete ids)"
        ),
    )
    chat.set_defaults(func=_cmd_chat)

    pull = sub.add_parser('pull', help='Download a model into the local cache')
    pull.add_argument('model', help='Alias or HF repo id (supports org/repo:quant)')
    pull.add_argument('--quant', default=None, help='Quantization variant (e.g. Q4_K_M)')
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

# Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Opt-in comparison of GenieX CLI JSON with llama.cpp's native endpoint.

Import the GGUF into GenieX first; pass the same GGUF to llama-server. Use
matching CPU/OpenCL builds (GGML_NATIVE=OFF); see README.md for commands.
"""

import argparse
import json
import math
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

STATE = 'I was charged twice this month, please refund one of them.'
QUESTIONS = {
    'refund': {'type': 'noul', 'instructions': 'Is the customer asking for a refund?'},
    'team': {
        'type': 'choice',
        'instructions': 'Which team should handle this?',
        'criteria': {
            'billing': 'Charges, refunds, invoices',
            'technical': 'App or site faults',
            'fraud': 'Suspected unauthorised use',
        },
    },
    'urgency': {
        'type': 'score',
        'instructions': 'How urgent is this?',
        'criteria': ['Can wait', 'Today', 'Blocking the customer now'],
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geniex', required=True, type=Path)
    parser.add_argument('--llama-server', required=True, type=Path)
    parser.add_argument('--gguf', required=True, type=Path)
    parser.add_argument('--model-name', required=True, help='Already-cached GenieX name:precision')
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--work-dir', type=Path, default=Path('.decision-comparison'))
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--compute', choices=['cpu', 'gpu'], default='cpu', help='CPU or OpenCL GPU offload')
    parser.add_argument('--tolerance', type=float, default=1e-5)
    args = parser.parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    cli = {}
    for name, question in QUESTIONS.items():
        cmd = [
            str(args.geniex.resolve()),
            'decide',
            args.model_name,
            '--data-dir',
            str(args.data_dir.resolve()),
            '--compute',
            args.compute,
            '--nctx',
            '4096',
            '--kind',
            'predicate' if question['type'] == 'noul' else question['type'],
            '--input',
            STATE,
            '--question',
            question['instructions'],
            '--json',
            '--log',
            'debug',
            '--skip-update',
        ]
        if question['type'] == 'choice':
            for key, value in question['criteria'].items():
                cmd += ['--choice', f'{key}={value}']
        if question['type'] == 'score':
            for label, description in zip(['low', 'medium', 'high'], question['criteria']):
                cmd += ['--level', f'{label}={description}']
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        # GGUF debug logs can truncate token strings inside a UTF-8 character.
        (args.work_dir / f'cli-{name}.log').write_bytes(result.stderr)
        result.check_returncode()
        cli[name] = json.loads(result.stdout)

    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    url = f'http://127.0.0.1:{port}'
    native = {}
    with (args.work_dir / 'llama-server.log').open('wb') as log:
        server = subprocess.Popen(
            [
                str(args.llama_server.resolve()),
                '-m',
                str(args.gguf.resolve()),
                '-ngl',
                '0' if args.compute == 'cpu' else '99',
                '-c',
                '4096',
                '-b',
                '4096',
                '-ub',
                '4096' if args.compute == 'cpu' else '512',
                '-t',
                str(args.threads),
                '-tb',
                str(args.threads),
                '-fa',
                'on' if args.compute == 'cpu' else 'off',
                '-np',
                '1',
                '--no-webui',
                '--cache-ram',
                '0',
                '--port',
                str(port),
                *(['--device', 'GPUOpenCL'] if args.compute == 'gpu' else []),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 120
            while True:
                if server.poll() is not None:
                    raise RuntimeError('llama-server exited; inspect llama-server.log')
                try:
                    with urllib.request.urlopen(url + '/health', timeout=1):
                        break
                except (OSError, urllib.error.URLError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('llama-server did not become healthy')
                    time.sleep(0.25)
            # GenieX deliberately evaluates isolated questions. One request per
            # question and one native slot avoid changing quantized computation
            # through shared-prefix reuse, padding, or multi-sequence batches.
            for name, question in QUESTIONS.items():
                request = urllib.request.Request(
                    url + '/v1/systemone',
                    data=json.dumps({'state': STATE, 'questions': {name: question}}).encode(),
                    headers={'Content-Type': 'application/json'},
                )
                with urllib.request.urlopen(request, timeout=120) as response:
                    result = json.load(response)
                if result['usage']['output_tokens'] != 0:
                    raise AssertionError('native decision unexpectedly generated tokens')
                native[name] = result['answers'][name]
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()

    differences = [abs(cli['refund']['probability'] - native['refund']['noul'])]
    for name in ['team', 'urgency']:
        probabilities = cli[name]['probabilities']
        if abs(sum(p['probability'] for p in probabilities) - 1) > 1e-6:
            raise AssertionError('GenieX probabilities do not sum to one')
        for probability in probabilities:
            differences.append(
                abs(probability['probability'] - native[name]['probabilities'][str(probability['value'])])
            )
    if cli['team']['choice'] != native['team']['choice']:
        raise AssertionError('choice mapping differs')
    # Preserve GenieX's expected-score contract. Native d1-2B defaults to a
    # median summary, so only the distribution is compared across endpoints.
    expected_score = sum(p['value'] * p['probability'] for p in cli['urgency']['probabilities'])
    if abs(cli['urgency']['score'] - expected_score) > 1e-6:
        raise AssertionError('GenieX score is not the expected level')
    maximum = max(differences)
    if not all(math.isfinite(difference) for difference in differences) or maximum > args.tolerance:
        raise AssertionError(f'maximum probability difference {maximum} exceeds {args.tolerance}')
    report = {
        'model': args.gguf.name,
        'backend': 'cpu' if args.compute == 'cpu' else 'gpu-opencl',
        'modality': 'text',
        'maximum_probability_difference': maximum,
        'cli': cli,
        'native': native,
    }
    (args.work_dir / 'comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(
        json.dumps({'model': args.gguf.name, 'backend': report['backend'], 'maximum_probability_difference': maximum})
    )


if __name__ == '__main__':
    main()

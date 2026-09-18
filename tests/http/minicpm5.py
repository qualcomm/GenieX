# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

"""Exercise MiniCPM5 through a running GenieX HTTP server (no mocks).

    geniex pull openbmb/MiniCPM5-1B-GGUF:Q4_K_M
    geniex serve --compute cpu --ngl 0
    python tests/http/minicpm5.py --model openbmb/MiniCPM5-1B-GGUF:Q4_K_M

Uses only the Python standard library. The model must already be available to
the server. --output saves the exact request/response bodies for review.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'get_weather',
            'description': 'Get the current weather for a city.',
            'parameters': {
                'type': 'object',
                'properties': {'city': {'type': 'string', 'description': 'The city name only, for example Beijing.'}},
                'required': ['city'],
            },
        },
    }
]
MESSAGES = [
    {
        'role': 'system',
        'content': 'Use get_weather once to answer weather questions. The city argument is a plain city name, '
        'not a JSON object. After receiving a tool result, answer the question using that result.',
    },
    {'role': 'user', 'content': 'What is the current weather in Beijing?'},
]


def request(args, name, body):
    req = urllib.request.Request(
        args.url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(body).encode(),
        headers={'Content-Type': 'application/json'},
    )
    try:
        response = urllib.request.urlopen(req, timeout=300)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        raw = response.read().decode()
        status = response.status
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / (name + '-request.json')).write_text(json.dumps(body, indent=2), encoding='utf-8')
        (args.output / (name + '-response.txt')).write_text(raw, encoding='utf-8')
    assert status == 200, (status, raw)
    return raw


def stream_message(raw):
    content, calls, finish = '', {}, None
    done = False
    for line in raw.splitlines():
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            done = True
            continue
        chunk = json.loads(data)
        assert 'error' not in chunk, chunk
        for choice in chunk.get('choices', []):
            delta = choice.get('delta', {})
            content += delta.get('content') or ''
            for part in delta.get('tool_calls', []):
                call = calls.setdefault(
                    part['index'],
                    {
                        'id': '',
                        'type': 'function',
                        'function': {'name': '', 'arguments': ''},
                    },
                )
                if part.get('id'):
                    call['id'] = part['id']
                for key in ('name', 'arguments'):
                    call['function'][key] += part.get('function', {}).get(key) or ''
            if choice.get('finish_reason'):
                finish = choice['finish_reason']
    assert done, 'missing SSE [DONE] terminator'
    assert list(sorted(calls)) == list(range(len(calls))), calls
    return {'role': 'assistant', 'content': content, 'tool_calls': list(calls.values())}, finish


def assert_weather_call(message, finish):
    assert finish == 'tool_calls', (finish, message)
    calls = message.get('tool_calls', [])
    assert len(calls) == 1, message
    call = calls[0]
    assert call.get('id') and call['type'] == 'function', call
    assert call['function']['name'] == 'get_weather', call
    assert json.loads(call['function']['arguments']) == {'city': 'Beijing'}, call
    content = message.get('content') or ''
    for marker in ('<function', '</function>', '<param', '</param>', 'name="get_weather"', 'name="city"'):
        assert marker not in content, message
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:18181')
    parser.add_argument('--model', required=True, help='Model name registered with the server')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = {
        'model': args.model,
        'compute': 'cpu',
        'ngl': 0,
        'nctx': 4096,
        # The SDK treats 0 as unset; negative temperature selects argmax.
        'temperature': -1,
        'seed': 42,
        'max_tokens': 256,
        'enable_think': False,
    }
    failures = []
    for stream in (False, True):
        name = 'streaming' if stream else 'blocking'
        try:
            messages = [dict(message) for message in MESSAGES]
            if stream:
                # Independent prompt/session, rather than replaying an already
                # completed prompt against a warm KV cache.
                messages[-1]['content'] = 'Please check the current weather in Beijing.'
            raw = request(args, name, dict(base, messages=messages, tools=TOOLS, stream=stream))
            if stream:
                message, finish = stream_message(raw)
            else:
                choice = json.loads(raw)['choices'][0]
                message, finish = choice['message'], choice['finish_reason']
            call = assert_weather_call(message, finish)
            print(f'PASS {name}: get_weather({call["function"]["arguments"]}), finish_reason={finish}', flush=True)

            # Feed the actual returned call ID and structured assistant message
            # back through template application, model inference, and HTTP.
            messages = messages + [
                message,
                {
                    'role': 'tool',
                    'tool_call_id': call['id'],
                    'content': '{"city":"Beijing","temperature_c":23,"condition":"sunny"}',
                },
            ]
            raw = request(args, name + '-followup', dict(base, messages=messages, tools=TOOLS, stream=stream))
            if stream:
                reply, finish = stream_message(raw)
            else:
                choice = json.loads(raw)['choices'][0]
                reply, finish = choice['message'], choice['finish_reason']
            assert finish == 'stop', (finish, reply)
            assert not reply.get('tool_calls'), reply
            assert '23' in (reply.get('content') or ''), reply
            print(f'PASS {name} tool-result round trip: {reply["content"]}', flush=True)
        except (AssertionError, KeyError, ValueError) as error:
            failures.append(name)
            print(f'FAIL {name}: {error}', flush=True)

    plain = json.loads(
        request(
            args,
            'plain',
            dict(
                base,
                messages=[
                    {
                        'role': 'user',
                        'content': 'Reply with the word Hello.',
                    }
                ],
            ),
        )
    )['choices'][0]
    assert plain['finish_reason'] == 'stop' and not plain['message'].get('tool_calls'), plain
    assert 'Hello' in plain['message']['content'], plain
    assert '<|' not in plain['message']['content'], plain
    print(f'PASS ordinary chat without tools: {plain["message"]["content"]}', flush=True)
    if failures:
        raise SystemExit('Failed: ' + ', '.join(failures))


if __name__ == '__main__':
    main()

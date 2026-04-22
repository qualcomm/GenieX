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

import platform
from typing import TypeAlias

from cases import *

PLUGIN_MAP = {
    'Linux': {
        'x86_64': ['llama_cpp', 'geniexml'],
        'arm64': ['llama_cpp'],
    },
    'Windows': {
        'x86_64': ['llama_cpp'],
        'arm64': ['llama_cpp', 'qairt', 'geniexml'],
    },
    'Darwin': {
        'x86_64': ['llama_cpp'],
        'arm64': ['llama_cpp', 'metal'],
    },
}

# (plugin, model_id, cases)
TESTCASE_MAP: dict[str, dict[str, dict[str, list[type[BaseCase]]]]] = {
    'llama_cpp': {
        'llm': {
            'Qwen/Qwen3-1.7B-GGUF:Q8_0': [MultiRound],
            # 'ggml-org/gemma-3-4b-it-GGUF:F16': [MultiRound, ImageMultiRound],
            'ggml-org/Qwen2.5-Omni-3B-GGUF:Q4_K_M': [MultiRound, AudioMultiRound],
        },
        'vlm': {},
        'embedder': {
            'djuna/jina-embeddings-v2-small-en-Q5_K_M-GGUF:Q5_K_M': [SingleRound],
        },
        'reranker': {},
        'tts': {},
        'asr': {},
        'diarize': {},
        'cv': {},
        'image_gen': {},
    },
    'qairt': {
        'llm': {
            'NexaAI/Granite-4-Micro-NPU': [MultiRound],
            'NexaAI/phi4-mini-npu-turbo': [MultiRound],
            'NexaAI/Qwen3-4B-Instruct-2507-npu': [MultiRound],
            'NexaAI/Qwen3-4B-Thinking-2507-npu': [MultiRound],
            'NexaAI/Llama3.2-3B-NPU-Turbo': [MultiRound],
            'NexaAI/jan-v1-4B-npu': [MultiRound],
            'NexaAI/qwen3-4B-npu': [MultiRound],
            'NexaAI/phi3.5-mini-npu': [MultiRound],
        },
        'vlm': {
            'NexaAI/Qwen3-VL-4B-Instruct-NPU': [MultiRound, ImageMultiRound],
            'NexaAI/OmniNeural-4B': [MultiRound, ImageMultiRound, AudioMultiRound],
            'NexaAI/LFM2-1.2B-npu': [MultiRound],
        },
        'embedder': {
            'NexaAI/embeddinggemma-300m-npu': [SingleRound],
        },
        'asr': {
            'NexaAI/parakeet-tdt-0.6b-v3-npu': [ASR],
        },
        'cv': {
            'NexaAI/convnext-tiny-npu': [ImageRecognition],
            'NexaAI/paddleocr-npu': [OCR],
            'NexaAI/yolov12-npu': [ImageRecognition],
        },
        'reranker': {
            'NexaAI/jina-v2-rerank-npu': [QueryDocument],
        },
    },
    'geniexml': {
        'vlm': {
            'NexaAI/Qwen3-VL-4B-Instruct-GGUF:Q4_0': [MultiRound, ImageMultiRound],
            'NexaAI/Qwen3-VL-4B-Thinking-GGUF:Q4_0': [MultiRound, ImageMultiRound],
        }
    },
}


def get_plugins() -> list[str]:
    system = platform.system()
    machine = platform.machine()
    return PLUGIN_MAP.get(system, {}).get(machine.lower(), [])


case: TypeAlias = tuple[str, str, str, list[type[BaseCase]]]


def get_testcases(plugins: list[str]) -> list[case]:
    res: list[case] = []
    for plugin in TESTCASE_MAP:
        if plugin not in plugins:
            continue
        for modal, model_cases in TESTCASE_MAP[plugin].items():
            for model, cases in model_cases.items():
                res.append((plugin, model, modal, cases))

    return res

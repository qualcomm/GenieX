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
    "Linux": {
        "x86_64": ["llama_cpp"],
        "arm64": ["llama_cpp", "qairt"],
    },
    "Windows": {
        "x86_64": ["llama_cpp"],
        "arm64": ["llama_cpp", "qairt"],
    },
}

# (plugin, model_id, cases)
TESTCASE_MAP: dict[str, dict[str, dict[str, list[type[BaseCase]]]]] = {
    "llama_cpp": {
        "llm": {
            "Qwen/Qwen3-1.7B-GGUF:Q8_0": [MultiRound],
        },
        "vlm": {
            "ggml-org/Qwen2.5-Omni-3B-GGUF:Q4_K_M": [MultiRound, AudioMultiRound],
            # 'ggml-org/gemma-3-4b-it-GGUF:F16': [MultiRound, ImageMultiRound],
        },
    },
    "qairt": {
        "llm": {
            "qualcomm/Qwen3-4B": [MultiRound],
        },
        "vlm": {
            "qualcomm/Qwen2.5-VL-7B-Instruct": [
                MultiRound,
                ImageMultiRound,
                AudioMultiRound,
            ],
        },
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

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

"""Unit coverage for the QAIRT n_ctx / n_gpu_layers coercion path in #763."""

from __future__ import annotations

import logging

from geniex.auto import PLUGIN_LLAMA_CPP, PLUGIN_QAIRT, _build_model_config


def test_qairt_coerces_user_n_gpu_layers_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger='geniex'):
        cfg = _build_model_config(PLUGIN_QAIRT, n_ctx=0, n_gpu_layers=64)
    assert cfg.n_gpu_layers == 0
    assert any('n_gpu_layers=64' in r.message for r in caplog.records)


def test_qairt_coerces_user_n_ctx_with_warning(caplog):
    with caplog.at_level(logging.WARNING, logger='geniex'):
        cfg = _build_model_config(PLUGIN_QAIRT, n_ctx=4096, n_gpu_layers=0)
    assert cfg.n_ctx == 0
    assert any('n_ctx=4096' in r.message for r in caplog.records)


def test_qairt_factory_defaults_pass_through_silently(caplog):
    with caplog.at_level(logging.WARNING, logger='geniex'):
        cfg = _build_model_config(PLUGIN_QAIRT, n_ctx=0, n_gpu_layers=999)
    assert cfg.n_gpu_layers == 0
    assert cfg.n_ctx == 0
    assert caplog.records == []


def test_llama_cpp_does_not_coerce_n_gpu_layers(caplog):
    with caplog.at_level(logging.WARNING, logger='geniex'):
        cfg = _build_model_config(PLUGIN_LLAMA_CPP, n_ctx=4096, n_gpu_layers=64)
    assert cfg.n_gpu_layers == 64
    assert cfg.n_ctx == 4096
    assert caplog.records == []

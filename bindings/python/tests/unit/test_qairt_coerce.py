# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

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
        cfg = _build_model_config(PLUGIN_QAIRT, n_ctx=0, n_gpu_layers=-1)
    assert cfg.n_gpu_layers == 0
    assert cfg.n_ctx == 0
    assert caplog.records == []


def test_llama_cpp_does_not_coerce_n_gpu_layers(caplog):
    with caplog.at_level(logging.WARNING, logger='geniex'):
        cfg = _build_model_config(PLUGIN_LLAMA_CPP, n_ctx=4096, n_gpu_layers=64)
    assert cfg.n_gpu_layers == 64
    assert cfg.n_ctx == 4096
    assert caplog.records == []

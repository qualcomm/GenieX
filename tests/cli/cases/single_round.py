# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import final, override

from .base import BaseCase


@final
class SingleRound(BaseCase):

    @override
    def param(self) -> list[str]:
        return [
            '-p',
            'hi, how are you today?',
        ]

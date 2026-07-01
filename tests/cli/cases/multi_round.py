# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any, final, override

from .base import BaseCase


@final
class MultiRound(BaseCase):

    def __init__(self) -> None:
        self.round = 0

    @override
    def param(self) -> list[str]:
        return ['-p', 'i am RemiliaForever', '-p', 'who am i', '-p', 'tell me a story']

    @override
    def check(self, res: Any) -> bool:
        self.round += 1
        return [
            lambda: True,  # round 1
            lambda: self.check_contain(res['Output']),  # round 2 
            lambda: True,  # round 3
        ][self.round - 1]()

    def check_contain(self, output: str) -> bool:
        return 'remilia' in output.lower()

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

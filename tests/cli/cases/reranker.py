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

from typing import final, override

from .base import BaseCase


@final
class QueryDocument(BaseCase):

    @override
    def param(self) -> list[str]:
        return [
            '-q',
            'What is the capital of France?',
            '-d',
            'Paris is the capital of France. It is known for its art, culture, and history.',
            '-d',
            'London is the capital of the United Kingdom. It is famous for its landmarks and museums.',
            '-d',
            'Berlin is the capital of Germany. It has a rich history and vibrant culture.',
        ]

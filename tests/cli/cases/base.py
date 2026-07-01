# Copyright 2024-2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Any


class BaseCase:

    def name(self) -> str:
        return self.__class__.__name__

    def param(self) -> list[str]:
        raise NotImplementedError

    def check(self, res: Any) -> bool:  # pyright: ignore[reportUnusedParameter]
        '''
        Optional check function to verify the result.
        '''
        return True

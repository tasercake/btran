"""Frozen Gate 1 boundary between orchestration and its callers."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias

from btran.config import Config


PageErrorCallback: TypeAlias = Callable[[int, str], None]
"""Called immediately as ``(page_number, error_message)`` on terminal page failure."""


@dataclass
class RunResult:
    """The completed-run result; an empty ``errors`` list means every page succeeded."""

    errors: list[str]


class OrchestratorCallable(Protocol):
    """Async runner signature: ``(config, on_page_error=None) -> RunResult``."""

    def __call__(
        self,
        config: Config,
        on_page_error: PageErrorCallback | None = None,
    ) -> Awaitable[RunResult]: ...

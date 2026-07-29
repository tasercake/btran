"""CLI finalization contract after immutable executor migration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.cli import main
from btran.config import Config
from btran.orchestrator_contract import RunResult


def _config(tmp_path: Path, **overrides: object) -> Config:
    values: dict[str, object] = {
        "input_dir": tmp_path / "input",
        "output_epub": tmp_path / "book.epub",
        "workspace": tmp_path / "work",
        "target_lang": "fr",
        "pi_bin": "definitely-not-present",
    }
    values.update(overrides)
    return Config(**values)


def test_cli_does_not_probe_model_or_epubcheck_before_runner(tmp_path):
    """Model/check executables belong to bounded stage invocation, not CLI."""
    config = _config(tmp_path, epub_check=True, epub_check_path="missing-epubcheck")
    runner = AsyncMock(return_value=RunResult(errors=[], status="completed"))
    with patch("btran.cli.load_config", return_value=config), patch("btran.cli.orchestrator_run", new=runner):
        main()
    runner.assert_awaited_once()


def test_cli_streams_recoverable_page_error_but_returns_zero(tmp_path, capsys):
    async def runner(config: Config, on_page_error=None) -> RunResult:
        assert on_page_error is not None
        on_page_error(3, "translation degraded; diagnostic content retained")
        return RunResult(errors=["legacy only"], status="completed_degraded")

    with patch("btran.cli.load_config", return_value=_config(tmp_path)), patch("btran.cli.orchestrator_run", new=runner):
        main()
    captured = capsys.readouterr()
    assert "page 3 failed" in captured.err
    assert "status=completed_degraded" in captured.out


def test_cli_keyboard_interrupt_has_terminal_nonzero_exit(tmp_path, capsys):
    async def runner(config: Config, on_page_error=None) -> RunResult:
        raise KeyboardInterrupt

    with patch("btran.cli.load_config", return_value=_config(tmp_path)), patch("btran.cli.orchestrator_run", new=runner):
        with pytest.raises(SystemExit) as exited:
            main()
    assert exited.value.code == 1
    assert "Interrupted" in capsys.readouterr().err

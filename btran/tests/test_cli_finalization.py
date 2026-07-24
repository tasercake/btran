"""CLI boundary tests; real orchestrator integration lives in a companion suite."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from btran.cli import main
from btran.config import Config
from btran.orchestrator_contract import RunResult


def _config(**overrides: object) -> Config:
    values: dict[str, object] = {
        "input_dir": Path("/tmp"),
        "output_epub": Path("/tmp/out.epub"),
        "target_lang": "fr",
        "pi_bin": "pi",
    }
    values.update(overrides)
    return Config(**values)


def test_fake_contract_success_returns_normally():
    async def fake_runner(config: Config, on_page_error=None) -> RunResult:
        assert config.target_lang == "fr"
        assert on_page_error is not None
        return RunResult(errors=[])

    with patch("btran.cli.load_config", return_value=_config()):
        with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
            with patch("btran.cli.orchestrator_run", new=fake_runner):
                main()


def test_fake_contract_failure_exits_nonzero_with_concise_summary(capsys):
    async def fake_runner(config: Config, on_page_error=None) -> RunResult:
        return RunResult(errors=["page 2 exhausted retries", "page 5 invalid response"])

    with patch("btran.cli.load_config", return_value=_config()):
        with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
            with patch("btran.cli.orchestrator_run", new=fake_runner):
                with pytest.raises(SystemExit) as exc:
                    main()

    assert exc.value.code == 1
    stderr = capsys.readouterr().err
    assert "2 page(s) failed" in stderr
    assert "page 2 exhausted retries" not in stderr
    assert "page 5 invalid response" not in stderr


def test_fake_contract_streams_page_error_before_completion(capsys):
    async def fake_runner(config: Config, on_page_error=None) -> RunResult:
        assert on_page_error is not None
        on_page_error(3, "network timeout")
        assert "page 3 failed: network timeout" in capsys.readouterr().err
        return RunResult(errors=["page 3 network timeout"])

    with patch("btran.cli.load_config", return_value=_config()):
        with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
            with patch("btran.cli.orchestrator_run", new=fake_runner):
                with pytest.raises(SystemExit) as exc:
                    main()

    assert exc.value.code == 1
    assert "1 page(s) failed" in capsys.readouterr().err


def test_input_path_must_be_a_directory(tmp_path, capsys):
    input_file = tmp_path / "not-a-directory"
    input_file.write_text("not images")

    with patch("btran.cli.load_config", return_value=_config(input_dir=input_file)):
        with pytest.raises(SystemExit) as exc:
            main()

    assert exc.value.code == 1
    assert "input_dir is not a directory" in capsys.readouterr().err


def test_epubcheck_executable_is_left_to_the_epub_stage_that_uses_it():
    config = _config(epub_check=True, epub_check_path="missing-epubcheck")
    called = False

    async def fake_runner(config: Config, on_page_error=None) -> RunResult:
        nonlocal called
        called = True
        return RunResult(errors=[])

    with patch("btran.cli.load_config", return_value=config):
        with patch(
            "btran.cli.shutil.which",
            side_effect=lambda executable: "/usr/bin/pi" if executable == "pi" else None,
        ):
            with patch("btran.cli.orchestrator_run", new=fake_runner):
                main()

    assert called is True

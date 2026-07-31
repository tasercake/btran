"""Task-16 CLI invocation-boundary acceptance coverage."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.cli import main, resolve_base_run_pointers
from btran.config import Config
from btran.manifest import InvocationFailure
from btran.orchestrator_contract import RunResult
from btran.schema import RunReport, StageRecord


def _config(tmp_path: Path, **overrides: object) -> Config:
    values: dict[str, object] = {
        "input_dir": tmp_path / "input",
        "output_epub": tmp_path / "book.epub",
        "workspace": tmp_path / "work",
        "target_lang": None,
    }
    values.update(overrides)
    return Config(**values)


def _result(status: str, *, failure: InvocationFailure | None = None) -> RunResult:
    return RunResult(errors=[], status=status, invocation_failure=failure)


def test_default_pointers_are_resolved_from_workspace(tmp_path):
    (tmp_path / "active-revision.json").write_text(json.dumps({"revision_id": "revision"}))
    (tmp_path / "active-correction-set.json").write_text(json.dumps({"set_id": "set"}))
    assert resolve_base_run_pointers(tmp_path, None, None) == ("revision", "set")
    assert resolve_base_run_pointers(tmp_path, "explicit", "other") == ("explicit", "other")


def test_cli_reports_modes_degraded_completion_and_stage_timings(tmp_path, capsys):
    config = _config(tmp_path)
    stage = StageRecord(stage="discovery", finding_ids=("summary",),
                        stage_summary_finding_id="summary", duration_ms=2.5)
    report = RunReport(run_id="run-1", final_epub_status="completed_degraded",
                       stage_records=(stage,), total_stage_duration_ms=2.5)
    runner = AsyncMock(return_value=RunResult(errors=[], status="completed_degraded", report=report))
    with patch("btran.cli.load_config", return_value=config), patch("btran.cli.orchestrator_run", new=runner):
        main()
    runner.assert_awaited_once()
    output = capsys.readouterr().out
    assert "mode=native" in output
    assert "status=completed_degraded" in output
    assert "btran timing_ms total=2.500 discovery=2.500" in output


def test_cli_invocation_failure_prints_typed_diagnostic_and_exits_one(tmp_path, capsys):
    config = _config(tmp_path)
    failure = InvocationFailure.input_access(config.input_dir, FileNotFoundError("missing"))
    with patch("btran.cli.load_config", return_value=config), patch(
        "btran.cli.orchestrator_run", new=AsyncMock(return_value=_result("invocation_failed", failure=failure)),
    ):
        with pytest.raises(SystemExit) as exited:
            main()
    assert exited.value.code == 1
    stderr = capsys.readouterr().err
    assert "invocation_failed code=input_access" in stderr
    assert "FileNotFoundError" in stderr


@pytest.mark.parametrize("kind", ["missing", "not_directory"])
def test_input_boundary_fails_before_any_model_call_and_writes_report(tmp_path, monkeypatch, capsys, kind):
    input_dir = tmp_path / "input"
    if kind == "not_directory":
        input_dir.write_text("not a directory", encoding="utf-8")
    output = tmp_path / "book.epub"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["btran", str(input_dir), str(output)])
    model_stage = AsyncMock()
    with patch("btran.orchestrator.extract_raw_pages", new=model_stage):
        with pytest.raises(SystemExit) as exited:
            main()
    assert exited.value.code == 1
    model_stage.assert_not_awaited()
    captured = capsys.readouterr()
    assert "invocation_failed code=input_access" in captured.err
    reports = list((tmp_path / ".btran" / "reports").glob("*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    assert report["invocation_failures"][0]["code"] == "input_access"

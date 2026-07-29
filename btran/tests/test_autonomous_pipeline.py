"""Task-16 autonomous completion and invocation-boundary acceptance tests."""

from __future__ import annotations

import os
import signal
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.artifacts import ArtifactStore
from btran.config import Config
from btran.epub_builder import EpubCheckError, EpubInvocationError, build_epub, check_epub, seal_effective_content
from btran.manifest import discover_book
from btran.orchestrator import run
from btran.schema import EffectivePage, EffectiveSegment, PageExtraction, SourceBlock, TermMention


def _config(tmp_path: Path) -> Config:
    return Config(input_dir=tmp_path / "input", output_epub=tmp_path / "book.epub",
                  workspace=tmp_path / "work", target_lang=None, timeout=1, max_retries=0)


def _image(path: Path) -> None:
    Image.new("RGB", (32, 32), "white").save(path)


def _source(path: Path, number: int) -> PageExtraction:
    return PageExtraction(number, str(path), "a" * 64, "b" * 16, "en", "test",
                          blocks=[SourceBlock(f"page_{number}_block", "paragraph", f"page {number}", 0)])


def _content():
    segment = EffectiveSegment(effective_segment_id="effective", segment_id="segment", source_lang="en",
                               source_text="content", effective_text="content", render_lang="en", mode="native")
    page = EffectivePage(effective_page_id="page", page_id="raw", effective_segment_ids=(segment.effective_segment_id,),
                         source_langs=("en",))
    return seal_effective_content((page,), (segment,))


@pytest.mark.asyncio
async def test_one_extract_failure_continues_to_complete_report_and_diagnostic_epub(tmp_path):
    config = _config(tmp_path)
    config.input_dir.mkdir()
    _image(config.input_dir / "one.png"); _image(config.input_dir / "two.png")

    async def extract(path, model, sha256, phash, page_number, **kwargs):
        if page_number == 1:
            raise RuntimeError("extract unavailable")
        return _source(Path(path), page_number)

    with patch("btran.source_extractor.extract_page", new=extract):
        result = await run(config)

    assert result.status == "completed"
    assert config.output_epub.is_file()
    assert result.report is not None and result.report_path
    assert {record.stage for record in result.report.stage_records} >= {
        "source_extraction", "reconciliation", "validation", "rendering", "candidate_seal",
    }
    assert any(record.status == "degraded" for record in result.report.stage_records)


@pytest.mark.asyncio
async def test_render_failure_still_seals_minimal_epub_and_complete_report(tmp_path):
    config = _config(tmp_path)
    config.input_dir.mkdir(); _image(config.input_dir / "page.png")
    with patch("btran.source_extractor.extract_page", return_value=_source(config.input_dir / "page.png", 1)), \
         patch("btran.epub_builder._rich_epub", side_effect=RuntimeError("renderer failed")):
        result = await run(config)
    assert result.status == "completed_degraded"
    assert config.output_epub.is_file()
    assert result.report is not None and result.report.final_epub_status == "completed_degraded"
    assert result.report.content_finding_ids


@pytest.mark.asyncio
@pytest.mark.parametrize("target_lang", [None, "fr"])
async def test_empty_readable_directory_renders_deterministic_diagnostic_without_model_calls(tmp_path, target_lang):
    config = _config(tmp_path)
    config.input_dir.mkdir()
    config = Config(**{**config.__dict__, "target_lang": target_lang})

    def model_called(*args, **kwargs):
        raise AssertionError("empty diagnostic must not call a model")

    with patch("btran.source_extractor.extract_page", side_effect=model_called), \
         patch("btran.orchestrator.make_pi_consolidation_call", side_effect=model_called), \
         patch("btran.translator.translate_segment", side_effect=model_called):
        first = await run(config)
    first_epub = config.output_epub.read_bytes()
    assert first.status == "completed_degraded"
    assert first.report is not None
    assert first.report.final_epub_status == "completed_degraded"
    assert first.report_path and Path(first.report_path).is_file()
    assert len({record.stage for record in first.report.stage_records}) == len(first.report.stage_records)
    assert {record.stage for record in first.report.stage_records} >= {
        "discovery", "source_extraction", "reconciliation", "validation", "rendering", "candidate_seal",
    }
    assert any(record.stage == "source_extraction" and record.status == "degraded" for record in first.report.stage_records)
    with zipfile.ZipFile(config.output_epub) as archive:
        chapters = [name for name in archive.namelist() if name.endswith(".xhtml")]
        assert chapters
        chapter = "\n".join(archive.read(name).decode("utf-8") for name in chapters)
    assert "No supported pages found" in chapter
    assert "no_supported_pages" in chapter

    with patch("btran.source_extractor.extract_page", side_effect=model_called), \
         patch("btran.orchestrator.make_pi_consolidation_call", side_effect=model_called), \
         patch("btran.translator.translate_segment", side_effect=model_called):
        second = await run(config)
    assert second.status == "completed_degraded"
    assert second.candidate_revision_id == first.candidate_revision_id
    assert config.output_epub.read_bytes() == first_epub


@pytest.mark.asyncio
@pytest.mark.parametrize("include_readable", [False, True])
async def test_explicit_target_completes_for_model_rejected_inputs_and_translation_failures(tmp_path, include_readable):
    config = _config(tmp_path)
    config = Config(**{**config.__dict__, "target_lang": "fr"})
    config.input_dir.mkdir()
    (config.input_dir / "bad.png").write_bytes(b"unreadable png")
    if include_readable:
        _image(config.input_dir / "good.png")

    async def extract(path, model, sha256, phash, page_number, **kwargs):
        if Path(path).read_bytes() == b"unreadable png":
            raise RuntimeError("vision rejected unreadable input")
        return _source(Path(path), page_number)

    translator = AsyncMock(side_effect=RuntimeError("translation unavailable"))
    with patch("btran.source_extractor.extract_page", new=extract), \
         patch("btran.translator.translate_segment", new=translator):
        result = await run(config)

    assert result.status == "completed_degraded"
    assert result.report is not None and result.report.final_epub_status == "completed_degraded"
    assert config.output_epub.is_file()
    assert translator.await_count == (1 if include_readable else 0)
    store = ArtifactStore(config.workspace)
    diagnostics = [store.get(artifact_id).payload for leaf in result.target_run.leaves
                   for artifact_id in leaf.segment_artifact_ids if store.get(artifact_id).payload["source_lang"] is None]
    assert diagnostics and all(item["mode"] == "translated" and item["render_lang"] == "und"
                               and item["translation_artifact_id"] is None for item in diagnostics)
    with zipfile.ZipFile(config.output_epub) as archive:
        package = next(archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(".opf"))
    assert "<dc:language>fr</dc:language>" in package


def _checker(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.mark.skipif(os.name != "posix", reason="process-session assertion requires POSIX")
def test_sleeping_epubcheck_is_bounded_and_kills_child(tmp_path):
    child_source = "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"
    program = _checker(tmp_path / "checker.py", (
        "import subprocess,sys,time\n"
        f"child=subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "open(sys.argv[1]+'.pid','w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    ))
    epub = tmp_path / "input.epub"
    started = time.monotonic()
    with pytest.raises(EpubCheckError, match="timed out"):
        check_epub(epub, str(program), timeout_seconds=1)
    assert time.monotonic() - started < 5
    child = int(Path(f"{epub}.pid").read_text())
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            break
        time.sleep(.01)
    else:
        os.kill(child, signal.SIGKILL)
        pytest.fail("EPUBCheck child survived bounded cleanup")


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process sessions")
async def test_terminology_timeout_kills_detached_pipe_holder_and_keeps_fallback_epub_report(tmp_path):
    """Escaped terminology child cannot hold cleanup pipes or block autonomous completion."""
    config = _config(tmp_path)
    config = Config(**{**config.__dict__, "target_lang": "fr", "pi_bin": str(tmp_path / "fake-pi")})
    config.input_dir.mkdir()
    _image(config.input_dir / "page.png")
    pid_path = tmp_path / "escaped-child.pid"
    child_source = (
        "import os, signal, time\n"
        "os.setsid()\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "while True: time.sleep(.1)\n"
    )
    Path(config.pi_bin).write_text(
        f"#!{sys.executable}\n"
        "import signal, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        f"open({str(pid_path)!r}, 'w').write(str(child.pid))\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True: time.sleep(.1)\n",
        encoding="utf-8",
    )
    Path(config.pi_bin).chmod(0o755)
    extraction = _source(config.input_dir / "page.png", 1)
    extraction.term_mentions = [TermMention("term", "page_1_block")]
    child_pid: int | None = None
    try:
        async def translate(*args, **kwargs):
            return "translated fallback"

        started = time.monotonic()
        with patch("btran.source_extractor.extract_page", return_value=extraction), \
             patch("btran.translator.translate_segment", new=translate):
            result = await run(config)
        assert time.monotonic() - started < 6
        assert pid_path.is_file()
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.01)
        else:
            pytest.fail("detached terminology pipe holder survived bounded cleanup")
        assert config.output_epub.is_file()
        assert result.report is not None and result.report_path
        store = ArtifactStore(config.workspace)
        findings = [store.get_finding(item) for item in result.report.content_finding_ids]
        assert any(item.kind == "terminology_consolidation_failed" for item in findings)
    finally:
        if child_pid is None and pid_path.exists():
            child_pid = int(pid_path.read_text(encoding="utf-8"))
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_staging_and_destination_errors_are_typed_invocation_errors(tmp_path, monkeypatch):
    content = _content()
    with patch("btran.epub_builder.tempfile.mkdtemp", side_effect=OSError("staging denied")):
        with pytest.raises(EpubInvocationError, match="staging"):
            build_epub(content, tmp_path / "book.epub")
    with patch("btran.epub_builder.os.replace", side_effect=OSError("destination denied")):
        with pytest.raises(EpubInvocationError, match="publish"):
            build_epub(content, tmp_path / "book.epub")


@pytest.mark.parametrize("failure", ["iteration", "generic_os", "page_stat", "page_read"])
def test_discovery_filesystem_errors_are_all_typed_input_access(tmp_path, monkeypatch, failure):
    input_dir = tmp_path / "input"; input_dir.mkdir()
    page = input_dir / "page.png"; page.write_bytes(b"bytes")
    if failure == "iteration":
        monkeypatch.setattr(Path, "iterdir", lambda self: (_ for _ in ()).throw(PermissionError("iteration denied")))
    elif failure == "generic_os":
        monkeypatch.setattr(Path, "iterdir", lambda self: (_ for _ in ()).throw(OSError("generic denied")))
    elif failure == "page_stat":
        original = Path.stat
        monkeypatch.setattr(Path, "stat", lambda self, *args, **kwargs: (_ for _ in ()).throw(PermissionError("stat denied")) if self == page else original(self, *args, **kwargs))
    else:
        original = Path.read_bytes
        monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(PermissionError("read denied")) if self == page else original(self))
    result = discover_book(input_dir)
    assert result.invocation_failure is not None
    assert result.invocation_failure.code == "input_access"

"""Task 12 renderer and bounded EPUBCheck tests."""

from __future__ import annotations

import os
import signal
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from xml.etree import ElementTree as ET

import pytest

from btran.artifacts import ArtifactStore
from btran.epub_builder import (
    EpubCheckError,
    SealedEffectiveContent,
    build_epub,
    check_epub,
    seal_effective_content,
)
from btran.schema import EffectivePage, EffectiveSegment


def segment(
    identifier: str, *, source_lang: str | None = "en", text: str = "Text", mode: str = "native",
    render_lang: str | None = None, translation_id: str | None = None,
) -> EffectiveSegment:
    return EffectiveSegment(
        effective_segment_id=f"effective-{identifier}", segment_id=f"segment-{identifier}",
        source_lang=source_lang, source_text=text if mode == "native" else f"source {text}",
        effective_text=text, render_lang=render_lang or (source_lang or "und"), mode=mode,
        translation_artifact_id=translation_id,
        finding_ids=("diagnostic-finding",) if source_lang is None else (),
    )


def page(identifier: str, *segments: EffectiveSegment) -> EffectivePage:
    return EffectivePage(
        effective_page_id=f"page-{identifier}", page_id=f"raw-page-{identifier}",
        effective_segment_ids=tuple(item.effective_segment_id for item in segments),
        source_langs=tuple(sorted({item.source_lang for item in segments if item.source_lang is not None})),
    )


def xhtml_files(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(".xhtml")}


def opf(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        return next(archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith(".opf"))


def checker(path: Path, source: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{source}", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_renderer_rejects_legacy_or_unsealed_input(tmp_path):
    with pytest.raises(TypeError, match="SealedEffectiveContent"):
        build_epub([], tmp_path / "book.epub")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="missing sealed segment"):
        SealedEffectiveContent((page("1", segment("one")),), ())


def test_native_segments_keep_their_own_languages_and_ordered_unique_metadata(tmp_path):
    japanese = segment("ja", source_lang="ja", text="日本語")
    english = segment("en", source_lang="en", text="English & <safe>")
    diagnostic = segment("diagnostic", source_lang=None, text="[diagnostic]", render_lang="und")
    content = seal_effective_content([page("1", japanese, english, diagnostic)], [japanese, english, diagnostic])
    output = tmp_path / "native.epub"

    result = build_epub(content, output, title="Native")

    assert result.status == "completed"
    chapter = next(value for name, value in xhtml_files(output).items() if "page_" in name)
    ET.fromstring(chapter)
    assert 'lang="ja" xml:lang="ja"' in chapter
    assert 'lang="en" xml:lang="en"' in chapter
    assert 'lang="und" xml:lang="und"' in chapter
    assert "English &amp; &lt;safe&gt;" in chapter
    package = opf(output)
    assert package.count("<dc:language>") == 2
    assert package.index("<dc:language>en</dc:language>") < package.index("<dc:language>ja</dc:language>")
    assert "und</dc:language>" not in package


def test_translated_rendering_has_target_language_and_source_provenance(tmp_path):
    translated = segment(
        "one", source_lang="ja", text="Hello", mode="translated", render_lang="fr",
        translation_id="translation-artifact",
    )
    content = seal_effective_content([page("1", translated)], [translated])
    output = tmp_path / "translated.epub"

    build_epub(content, output)

    chapter = next(value for name, value in xhtml_files(output).items() if "page_" in name)
    assert 'lang="fr" xml:lang="fr"' in chapter
    assert 'data-source-lang="ja"' in chapter
    assert 'data-translation-artifact-id="translation-artifact"' in chapter
    assert "Hello" in chapter
    assert "<dc:language>fr</dc:language>" in opf(output)


def test_translated_document_allows_und_diagnostic_segments_and_keeps_target_metadata(tmp_path):
    translated = segment("one", source_lang="ja", text="Bonjour", mode="translated", render_lang="fr",
                         translation_id="translation-artifact")
    diagnostic = segment("diagnostic", source_lang=None, text="[bad & <input>]", mode="translated", render_lang="und")
    translated_page = page("translated", translated)
    diagnostic_page = EffectivePage(
        effective_page_id="page-diagnostic", page_id="raw-page-diagnostic",
        effective_segment_ids=(diagnostic.effective_segment_id,), source_langs=(),
        display_metadata={"target_lang": "fr"},
    )
    content = seal_effective_content((translated_page, diagnostic_page), (translated, diagnostic))
    output = tmp_path / "mixed-target-diagnostic.epub"

    build_epub(content, output)

    chapters = xhtml_files(output)
    normal = next(value for value in chapters.values() if "Bonjour" in value)
    diagnostic_chapter = next(value for value in chapters.values() if "bad &amp; &lt;input&gt;" in value)
    assert 'lang="fr" xml:lang="fr"' in normal
    assert 'lang="und" xml:lang="und"' in diagnostic_chapter
    assert '<dc:language>fr</dc:language>' in opf(output)


def test_closed_input_rejects_bad_page_language_or_mixed_targets():
    one = segment("one", source_lang="en")
    bad_page = EffectivePage(
        effective_page_id="page", page_id="raw", effective_segment_ids=(one.effective_segment_id,), source_langs=("fr",),
    )
    with pytest.raises(ValueError, match="source languages"):
        seal_effective_content([bad_page], [one])
    french = segment("fr", source_lang="en", mode="translated", render_lang="fr")
    german = segment("de", source_lang="en", mode="translated", render_lang="de")
    with pytest.raises(ValueError, match="one target"):
        seal_effective_content([page("x", french, german)], [french, german])
    with pytest.raises(ValueError, match="cannot mix native and translated"):
        seal_effective_content([page("native", one), page("translated", french)], [one, french])


def test_epubcheck_nonzero_has_bounded_unicode_tails(tmp_path):
    program = checker(
        tmp_path / "check.py",
        "import sys\nsys.stdout.write('x' * 9000)\nsys.stderr.write('é' * 9000)\nsys.exit(3)\n",
    )
    with pytest.raises(EpubCheckError) as raised:
        check_epub(tmp_path / "input.epub", str(program), timeout_seconds=1)
    error = raised.value
    assert error.returncode == 3
    assert error.stdout_tail.endswith("…[truncated]")
    assert error.stderr_tail.endswith("…[truncated]")
    assert len(error.stdout_tail) == 8192 + len("…[truncated]")
    assert len(error.stderr_tail) == 8192 + len("…[truncated]")


@pytest.mark.parametrize("value", [0, -1, 3601, float("nan"), float("inf"), float("-inf"), True, 1.5, "1"])
@pytest.mark.parametrize("argument", ["timeout_seconds", "timeout"])
def test_epubcheck_rejects_invalid_config_timeout_before_spawning(tmp_path, value, argument):
    with patch("btran.epub_builder.subprocess.Popen") as popen:
        with pytest.raises(ValueError, match="timeout must be an integer between 1 and 3600"):
            check_epub(tmp_path / "input.epub", **{argument: value})  # type: ignore[arg-type]
    popen.assert_not_called()


@pytest.mark.parametrize(("argument", "value"), [("timeout_seconds", 1), ("timeout", 3600)])
def test_epubcheck_accepts_config_timeout_boundaries(tmp_path, argument, value):
    proc = MagicMock()
    proc.communicate.return_value = ("", "")
    proc.returncode = 0
    with patch("btran.epub_builder.subprocess.Popen", return_value=proc) as popen:
        check_epub(tmp_path / "input.epub", **{argument: value})
    popen.assert_called_once()
    proc.communicate.assert_called_once_with(timeout=value)


def test_epubcheck_timeout_terminates_bounded_process_session(tmp_path):
    program = checker(tmp_path / "sleep.py", "import time\ntime.sleep(60)\n")
    start = time.monotonic()
    with pytest.raises(EpubCheckError, match="timed out"):
        check_epub(tmp_path / "input.epub", str(program), timeout_seconds=1)
    assert time.monotonic() - start < 5


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_epubcheck_timeout_kills_term_ignoring_child_after_parent_exits(tmp_path):
    child_source = "import os, signal, time\nos.setsid()\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\nwhile True: time.sleep(.1)\n"
    program = checker(
        tmp_path / "parent.py",
        "import signal, subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "with open(sys.argv[1] + '.child-pid', 'w') as handle:\n"
        "    handle.write(str(child.pid))\n"
        "    handle.flush()\n"
        "signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))\n"
        "while True: time.sleep(.1)\n",
    )
    input_path = tmp_path / "input.epub"
    pid_path = Path(f"{input_path}.child-pid")
    child_pid: int | None = None
    try:
        with pytest.raises(EpubCheckError, match="timed out"):
            check_epub(input_path, str(program), timeout_seconds=1)
        child_pid = int(pid_path.read_text(encoding="utf-8"))
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(child_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(.01)
        else:
            pytest.fail("TERM-ignoring EPUBCheck child survived group cleanup")
    finally:
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_check_only_failure_persists_finding_and_keeps_rich_epub(tmp_path):
    item = segment("one", text="content")
    content = seal_effective_content([page("1", item)], [item])
    output = tmp_path / "checked.epub"
    store = ArtifactStore(tmp_path / "state")
    program = checker(tmp_path / "fail.py", "import sys\nsys.stderr.write('bad check')\nsys.exit(1)\n")

    result = build_epub(content, output, epub_check=True, epub_check_path=str(program), timeout_seconds=1, artifact_store=store)

    assert result.status == "completed_degraded"
    assert result.rich_epub_retained is True
    finding = store.get_finding(result.finding_ids[0])
    assert finding.kind == "epubcheck_failed"
    assert finding.audit_category == "fallback"
    assert finding.evidence["trigger"] == "epubcheck_failed:EpubCheckError"
    assert any("page_" in name for name in xhtml_files(output))


def test_rich_failure_persists_finding_and_publishes_deterministic_minimal_epub(tmp_path, monkeypatch):
    item = segment("one", text="Bad & <escaped>")
    content = seal_effective_content([page("1", item)], [item])
    store = ArtifactStore(tmp_path / "state")
    output = tmp_path / "fallback.epub"

    def fail(*args, **kwargs):
        raise RuntimeError("template broke")

    monkeypatch.setattr("btran.epub_builder._rich_epub", fail)
    result = build_epub(content, output, artifact_store=store)

    assert result.status == "completed_degraded"
    finding = store.get_finding(result.finding_ids[0])
    assert finding.kind == "render_failed"
    assert finding.audit_category == "fallback"
    assert finding.evidence["trigger"] == "render_failed:RuntimeError"
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist()[0] == "mimetype"
        diagnostic = archive.read("OEBPS/text/diagnostic.xhtml").decode("utf-8")
    assert "Bad &amp; &lt;escaped&gt;" in diagnostic
    assert 'lang="en" xml:lang="en"' in diagnostic


def test_minimal_diagnostic_epub_is_deterministic(tmp_path, monkeypatch):
    item = segment("one", text="same")
    content = seal_effective_content([page("1", item)], [item])
    monkeypatch.setattr("btran.epub_builder._rich_epub", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("x")))
    first, second = tmp_path / "first.epub", tmp_path / "second.epub"
    build_epub(content, first)
    build_epub(content, second)
    assert first.read_bytes() == second.read_bytes()

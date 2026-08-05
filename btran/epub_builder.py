"""Render sealed effective content and run bounded EPUBCheck sessions."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Mapping, Sequence
import zipfile

from ebooklib import epub

from btran.artifacts import ArtifactStore
from btran.config import validate_timeout_seconds
from btran.process_cleanup import CleanupCause, cleanup_popen
from btran.schema import EffectivePage, EffectiveSegment, Finding, canonical_json, tagged_sha256


CSS = """
body { font-family: Georgia, serif; line-height: 1.6; margin: 1em auto; max-width: 42em; }
section.page { margin-bottom: 2em; }
p.segment { white-space: pre-wrap; }
"""
_TAIL_LIMIT = 8_192
_CLEANUP_SECONDS = 2


class EpubCheckError(RuntimeError):
    """EPUBCheck did not complete successfully within its bounded session."""

    def __init__(
        self, message: str, *, executable: str, timeout_seconds: float,
        returncode: int | None = None, stdout_tail: str = "", stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.executable = executable
        self.timeout_seconds = timeout_seconds
        self.returncode = returncode
        self.stdout_tail = stdout_tail
        self.stderr_tail = stderr_tail


class EpubInvocationError(OSError):
    """Staging, bundle, or destination filesystem access failed."""


@dataclass(frozen=True)
class RenderPlacement:
    """One ordered physical output position for an immutable logical page."""

    placement_id: str
    page_id: str
    effective_page_id: str
    relative_path: str = ""

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) and value for value in
                   (self.placement_id, self.page_id, self.effective_page_id)):
            raise ValueError("render placement needs placement, logical-page, and effective-page IDs")
        if not isinstance(self.relative_path, str):
            raise ValueError("render placement relative path must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"placement_id": self.placement_id, "page_id": self.page_id,
                "effective_page_id": self.effective_page_id, "relative_path": self.relative_path}


@dataclass(frozen=True)
class SealedEffectiveContent:
    """Closed renderer input.  No legacy extraction/translation records are accepted."""

    pages: tuple[EffectivePage, ...]
    segments: tuple[EffectiveSegment, ...]
    placements: tuple[RenderPlacement, ...] = ()

    def __post_init__(self) -> None:
        if not self.pages:
            raise ValueError("sealed effective content must contain at least one page")
        if not all(isinstance(page, EffectivePage) for page in self.pages):
            raise TypeError("renderer accepts only EffectivePage records")
        if not all(isinstance(segment, EffectiveSegment) for segment in self.segments):
            raise TypeError("renderer accepts only EffectiveSegment records")
        page_ids = [page.effective_page_id for page in self.pages]
        if len(set(page_ids)) != len(page_ids):
            raise ValueError("effective page IDs must be unique")
        segment_by_id = {segment.effective_segment_id: segment for segment in self.segments}
        if len(segment_by_id) != len(self.segments):
            raise ValueError("effective segment IDs must be unique")
        used: list[str] = []
        modes: set[str] = set()
        for page in self.pages:
            if len(set(page.effective_segment_ids)) != len(page.effective_segment_ids):
                raise ValueError("effective page segment IDs must be unique")
            page_segments: list[EffectiveSegment] = []
            for segment_id in page.effective_segment_ids:
                try:
                    segment = segment_by_id[segment_id]
                except KeyError as exc:
                    raise ValueError("effective page names a missing sealed segment") from exc
                page_segments.append(segment)
                used.append(segment_id)
                modes.add(segment.mode)
            expected_langs = tuple(sorted({segment.source_lang for segment in page_segments if segment.source_lang is not None}))
            if page.source_langs != expected_langs:
                raise ValueError("effective page source languages do not match sealed segments")
        if len(used) != len(set(used)) or set(used) != set(segment_by_id):
            raise ValueError("every sealed segment must belong to exactly one effective page")
        placements = self.placements or tuple(
            RenderPlacement(f"logical-{page.effective_page_id}", page.page_id, page.effective_page_id)
            for page in self.pages
        )
        if not all(isinstance(placement, RenderPlacement) for placement in placements):
            raise TypeError("renderer accepts only RenderPlacement records")
        if len({placement.placement_id for placement in placements}) != len(placements):
            raise ValueError("render placement IDs must be unique")
        page_by_id = {page.page_id: page for page in self.pages}
        if any(page_by_id.get(placement.page_id) is None
               or page_by_id[placement.page_id].effective_page_id != placement.effective_page_id
               for placement in placements):
            raise ValueError("render placement must reference its sealed logical page")
        if set(placement.page_id for placement in placements) != set(page_by_id):
            raise ValueError("every sealed logical page needs a render placement")
        object.__setattr__(self, "placements", tuple(placements))
        if len(modes) > 1:
            raise ValueError("sealed effective content cannot mix native and translated segments")
        if modes == {"translated"}:
            # Source/extraction diagnostics are target-document leaves but are
            # intentionally language-undetermined. Ordinary translated leaves
            # must still agree on exactly one target language.
            target_languages = {segment.render_lang for segment in self.segments if segment.source_lang is not None}
            declared_languages = {
                page.display_metadata["target_lang"] for page in self.pages
                if "target_lang" in page.display_metadata
            }
            if any(not isinstance(language, str) or not language or language == "und" for language in declared_languages):
                raise ValueError("translated sealed content has invalid target metadata language")
            if len(target_languages) > 1 or len(declared_languages) > 1:
                raise ValueError("translated sealed content must have one target render language")
            if target_languages and declared_languages and target_languages != declared_languages:
                raise ValueError("translated sealed content target metadata does not match render language")
            if any(segment.source_lang is None and segment.render_lang != "und" for segment in self.segments):
                raise ValueError("translated diagnostic segments must render as und")

    @classmethod
    def from_records(
        cls, pages: Sequence[EffectivePage], segments: Sequence[EffectiveSegment] | Mapping[str, EffectiveSegment],
        placements: Sequence[RenderPlacement] = (),
    ) -> "SealedEffectiveContent":
        if isinstance(segments, Mapping):
            values = tuple(segments.values())
        else:
            values = tuple(segments)
        return cls(tuple(pages), values, tuple(placements))

    @property
    def mode(self) -> str:
        return self.segments[0].mode if self.segments else "native"

    @property
    def target_lang(self) -> str:
        if self.mode != "translated":
            return "und"
        ordinary = {segment.render_lang for segment in self.segments if segment.source_lang is not None}
        if ordinary:
            return next(iter(ordinary))
        declared = {page.display_metadata["target_lang"] for page in self.pages if "target_lang" in page.display_metadata}
        return next(iter(declared)) if declared else "und"

    def page_segments(self, page: EffectivePage) -> tuple[EffectiveSegment, ...]:
        by_id = {segment.effective_segment_id: segment for segment in self.segments}
        return tuple(by_id[segment_id] for segment_id in page.effective_segment_ids)


def seal_effective_content(
    pages: Sequence[EffectivePage], segments: Sequence[EffectiveSegment] | Mapping[str, EffectiveSegment],
    placements: Sequence[RenderPlacement] = (),
) -> SealedEffectiveContent:
    """Make the only renderer input accepted by :func:`build_epub`."""
    return SealedEffectiveContent.from_records(pages, segments, placements)


def _xml_safe(value: str) -> str:
    return "".join(
        char if ord(char) in {0x9, 0xA, 0xD} or 0x20 <= ord(char) <= 0xD7FF
        or 0xE000 <= ord(char) <= 0xFFFD or 0x10000 <= ord(char) <= 0x10FFFF else "\ufffd"
        for char in value
    )


def _escaped(value: str) -> str:
    return escape(_xml_safe(value), quote=True)


def _tail(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    if len(value) <= _TAIL_LIMIT:
        return value
    return value[-_TAIL_LIMIT:] + "…[truncated]"


def _timed_out_cleanup(proc: subprocess.Popen[str]) -> tuple[str, str]:
    """Shared TERM/KILL cleanup includes detached inherited-pipe descendants."""
    return cleanup_popen(
        proc, cause=CleanupCause.FAILURE, term_grace=_CLEANUP_SECONDS, kill_grace=_CLEANUP_SECONDS,
    )


def check_epub(
    path: Path, executable: str = "epubcheck", timeout_seconds: int = 120, *, timeout: int | None = None,
) -> None:
    """Run EPUBCheck in its own process session with Config-bounded timeout."""
    timeout_seconds = validate_timeout_seconds(timeout if timeout is not None else timeout_seconds)
    try:
        proc = subprocess.Popen(
            [executable, str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            start_new_session=True,
        )
    except OSError as exc:
        raise EpubCheckError(
            f"EPUBCheck spawn failed executable={executable!r}: {exc}", executable=executable,
            timeout_seconds=float(timeout_seconds), stderr_tail=_tail(str(exc)),
        ) from exc
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        stdout, stderr = _timed_out_cleanup(proc)
        # communicate's partial data is useful when a mocked wrapper does not
        # return it after cleanup.
        stdout = stdout or _tail(exc.output)
        stderr = stderr or _tail(exc.stderr)
        raise EpubCheckError(
            f"EPUBCheck timed out executable={executable!r} timeout={timeout_seconds}s "
            f"exit={proc.returncode!r} stdout_tail={_tail(stdout)!r} stderr_tail={_tail(stderr)!r}",
            executable=executable, timeout_seconds=float(timeout_seconds), returncode=proc.returncode,
            stdout_tail=_tail(stdout), stderr_tail=_tail(stderr),
        ) from None
    stdout_tail, stderr_tail = _tail(stdout), _tail(stderr)
    if proc.returncode:
        raise EpubCheckError(
            f"EPUBCheck failed executable={executable!r} timeout={timeout_seconds}s exit={proc.returncode} "
            f"stdout_tail={stdout_tail!r} stderr_tail={stderr_tail!r}", executable=executable,
            timeout_seconds=float(timeout_seconds), returncode=proc.returncode,
            stdout_tail=stdout_tail, stderr_tail=stderr_tail,
        )


def _safe_anchor(value: str, fallback: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "_.-" else "-" for char in value).strip("-.")
    return cleaned or fallback


def _page_xhtml(page: EffectivePage, placement: RenderPlacement, segments: Sequence[EffectiveSegment], title: str) -> str:
    body: list[str] = [
        f'<section class="page" id="{_escaped(_safe_anchor(placement.placement_id, "page"))}" '
        f'data-placement-id="{_escaped(placement.placement_id)}" data-page-id="{_escaped(page.page_id)}" '
        f'data-effective-page-id="{_escaped(page.effective_page_id)}" data-relative-path="{_escaped(placement.relative_path)}">'
    ]
    for position, segment in enumerate(segments, start=1):
        attrs = {
            "id": _safe_anchor(segment.effective_segment_id, f"segment-{position}"),
            "class": "segment",
            "lang": segment.render_lang,
            "xml:lang": segment.render_lang,
            "data-segment-id": segment.segment_id,
            "data-source-lang": segment.source_lang or "und",
            "data-mode": segment.mode,
        }
        if segment.translation_artifact_id is not None:
            attrs["data-translation-artifact-id"] = segment.translation_artifact_id
        if segment.source_overlay_artifact_id is not None:
            attrs["data-source-overlay-artifact-id"] = segment.source_overlay_artifact_id
        if segment.target_overlay_artifact_id is not None:
            attrs["data-target-overlay-artifact-id"] = segment.target_overlay_artifact_id
        rendered_attrs = " ".join(f'{name}="{_escaped(value)}"' for name, value in attrs.items())
        body.append(f"<p {rendered_attrs}>{_escaped(segment.effective_text)}</p>")
    body.append("</section>")
    document_lang = segments[0].render_lang if segments else "und"
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
        f'xml:lang="{_escaped(document_lang)}" lang="{_escaped(document_lang)}">\n'
        f"<head><title>{_escaped(title)}</title><link rel=\"stylesheet\" type=\"text/css\" href=\"../style/default.css\"/></head>\n"
        '<body epub:type="bodymatter" role="document">\n' + "\n".join(body) + "\n</body>\n</html>"
    )


def _source_languages(content: SealedEffectiveContent) -> tuple[str, ...]:
    languages: list[str] = []
    for page in content.pages:
        for language in page.source_langs:
            if language not in languages:
                languages.append(language)
    return tuple(languages)


def _rich_epub(content: SealedEffectiveContent, path: Path, *, title: str, author: str) -> None:
    book = epub.EpubBook()
    content_id = tagged_sha256("rendered-effective-content-v1", canonical_json({
        "pages": [page.to_dict() for page in content.pages],
        "segments": [segment.to_dict() for segment in content.segments],
        "placements": [placement.to_dict() for placement in content.placements],
    }).encode("utf-8"))
    book.set_identifier(f"btran-{content_id}")
    book.set_title(_xml_safe(title))
    book.add_author(_xml_safe(author))
    languages = _source_languages(content) if content.mode == "native" else (content.target_lang,)
    languages = languages or ("und",)
    book.set_language(languages[0])
    for language in languages[1:]:
        book.add_metadata("DC", "language", language)
    book.add_item(epub.EpubItem(uid="style", file_name="style/default.css", media_type="text/css", content=CSS.encode("utf-8")))
    chapters: list[epub.EpubHtml] = []
    page_by_id = {page.page_id: page for page in content.pages}
    for number, placement in enumerate(content.placements, start=1):
        page = page_by_id[placement.page_id]
        chapter = epub.EpubHtml(
            uid=f"placement-{_safe_anchor(placement.placement_id, str(number))}", file_name=f"text/page_{number:04}.xhtml", title=f"Page {number}",
            lang=(content.page_segments(page)[0].render_lang if page.effective_segment_ids else "und"),
        )
        chapter.content = _page_xhtml(page, placement, content.page_segments(page), f"{title} — Page {number}").encode("utf-8")
        book.add_item(chapter)
        chapters.append(chapter)
    book.toc = tuple(chapters)
    book.spine = ["nav", *chapters]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav(title="Contents"))
    book.guide = [{"type": "text", "item": chapters[0]}]
    epub.write_epub(str(path), book, {"epub3_landmark": True})


def _minimal_xhtml(content: SealedEffectiveContent, title: str) -> str:
    parts = ["<p>Diagnostic EPUB: rich rendering failed; effective content follows.</p>"]
    page_by_id = {page.page_id: page for page in content.pages}
    for placement in content.placements:
        page = page_by_id[placement.page_id]
        parts.append(f'<section data-placement-id="{_escaped(placement.placement_id)}" data-page-id="{_escaped(page.page_id)}" data-effective-page-id="{_escaped(page.effective_page_id)}">')
        for segment in content.page_segments(page):
            parts.append(
                f'<p lang="{_escaped(segment.render_lang)}" xml:lang="{_escaped(segment.render_lang)}" '
                f'data-segment-id="{_escaped(segment.segment_id)}">{_escaped(segment.effective_text)}</p>'
            )
        parts.append("</section>")
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="und" lang="und"><head><title>'
        + _escaped(title) + "</title></head><body>" + "".join(parts) + "</body></html>"
    )


def _zip_entry(archive: zipfile.ZipFile, name: str, content: bytes, *, stored: bool = False) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED if stored else zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def _minimal_epub(content: SealedEffectiveContent, path: Path, *, title: str, author: str) -> None:
    identifier = tagged_sha256("minimal-diagnostic-epub-v1", canonical_json({
        "pages": [page.to_dict() for page in content.pages], "segments": [segment.to_dict() for segment in content.segments],
        "placements": [placement.to_dict() for placement in content.placements], "title": title, "author": author,
    }).encode("utf-8"))
    xhtml = _minimal_xhtml(content, title).encode("utf-8")
    opf = (
        '<?xml version="1.0" encoding="utf-8"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
        'unique-identifier="bookid"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:identifier id="bookid">'
        + _escaped(identifier) + "</dc:identifier><dc:title>" + _escaped(title) + "</dc:title><dc:creator>"
        + _escaped(author) + '</dc:creator><dc:language>' + _escaped(content.target_lang) + '</dc:language></metadata><manifest><item id="diagnostic" '
        'href="text/diagnostic.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="diagnostic"/>'
        "</spine></package>"
    ).encode("utf-8")
    container = b'<?xml version="1.0" encoding="UTF-8"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>'
    with zipfile.ZipFile(path, "w") as archive:
        _zip_entry(archive, "mimetype", b"application/epub+zip", stored=True)
        _zip_entry(archive, "META-INF/container.xml", container)
        _zip_entry(archive, "OEBPS/content.opf", opf)
        _zip_entry(archive, "OEBPS/text/diagnostic.xhtml", xhtml)


def _finding(kind: str, content: SealedEffectiveContent, error: Exception) -> Finding:
    """Describe a non-blocking rich-output degradation for the final audit."""
    return Finding(
        kind=kind, severity="warning", stage="rendering",
        subject_refs=tuple(sorted(page.effective_page_id for page in content.pages)),
        evidence={
            "trigger": f"{kind}:{type(error).__name__}",
            "exception_type": type(error).__name__,
            "message": _xml_safe(str(error)),
        },
        audit_category="fallback",
        message=(
            "EPUB rich rendering failed; deterministic minimal EPUB fallback was produced."
            if kind == "render_failed" else
            "EPUBCheck failed; rich EPUB was retained and the diagnostic fallback was produced."
        ),
    )


def _persist(finding: Finding, store: ArtifactStore | None, sink: Callable[[Finding], object] | None) -> str:
    if store is not None:
        return store.put_finding(finding)
    if sink is not None:
        sink(finding)
    return finding.finding_id


@dataclass(frozen=True)
class EpubBuildResult:
    output_path: Path
    status: str
    finding_ids: tuple[str, ...] = ()
    rich_epub_retained: bool = False


def build_epub(
    content: SealedEffectiveContent, output_path: Path, *, title: str = "Translated Book", author: str = "Unknown",
    epub_check: bool = False, epub_check_path: str = "epubcheck", timeout_seconds: float = 120,
    artifact_store: ArtifactStore | None = None, finding_sink: Callable[[Finding], object] | None = None,
    timeout: float | None = None,
) -> EpubBuildResult:
    """Stage rich output, then publish rich or deterministic diagnostic EPUB.

    ``content`` must already be closed/sealed.  Renderer/check failures are
    content findings; only filesystem access while staging/publishing raises
    :class:`EpubInvocationError`.
    """
    if not isinstance(content, SealedEffectiveContent):
        raise TypeError("build_epub accepts only SealedEffectiveContent")
    if timeout is not None:
        timeout_seconds = timeout
    output_path = Path(output_path)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stage_dir = Path(tempfile.mkdtemp(prefix=".btran-epub-", dir=output_path.parent))
    except OSError as exc:
        raise EpubInvocationError(f"unable to create EPUB staging area: {exc}") from exc
    rich_path = stage_dir / "rich.epub"
    minimal_path = stage_dir / "diagnostic.epub"
    finding_ids: list[str] = []
    status = "completed"
    keep_rich = False
    try:
        try:
            _rich_epub(content, rich_path, title=title, author=author)
        except OSError as exc:
            raise EpubInvocationError(f"unable to write staged rich EPUB: {exc}") from exc
        except Exception as exc:
            status = "completed_degraded"
            finding_ids.append(_persist(_finding("render_failed", content, exc), artifact_store, finding_sink))
            try:
                _minimal_epub(content, minimal_path, title=title, author=author)
            except OSError as write_error:
                raise EpubInvocationError(f"unable to write staged diagnostic EPUB: {write_error}") from write_error
            publish = minimal_path
        else:
            publish = rich_path
            if epub_check:
                try:
                    check_epub(rich_path, epub_check_path, timeout_seconds)
                except EpubCheckError as exc:
                    status = "completed_degraded"
                    keep_rich = True
                    finding_ids.append(_persist(_finding("epubcheck_failed", content, exc), artifact_store, finding_sink))
                    # Produce the independent deterministic diagnostic artifact
                    # for inspection, but publish rich content on check-only failure.
                    try:
                        _minimal_epub(content, minimal_path, title=title, author=author)
                    except OSError as write_error:
                        raise EpubInvocationError(f"unable to write staged diagnostic EPUB: {write_error}") from write_error
        try:
            os.replace(publish, output_path)
        except OSError as exc:
            raise EpubInvocationError(f"unable to publish EPUB output: {exc}") from exc
        return EpubBuildResult(output_path, status, tuple(finding_ids), keep_rich)
    finally:
        try:
            for child in stage_dir.iterdir():
                child.unlink(missing_ok=True)
            stage_dir.rmdir()
        except OSError:
            # Staging cleanup cannot change an already published content result.
            pass

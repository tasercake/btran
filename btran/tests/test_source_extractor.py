"""Tests for typed source extraction from page images."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.schema import PageExtraction, SourceBlock, TermMention


def _make_mock_proc(stdout: str = "", stderr: str = "", returncode: int = 0):
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    return proc


_VALID_OUTPUT = json.dumps(
    {
        "blocks": [
            {"id": "scan-heading", "type": "heading", "text": "Chapter One", "reading_order": 0},
            {"id": "scan-body", "type": "paragraph", "text": "The first paragraph.", "reading_order": 1},
            {"id": "scan-figure", "type": "illustration", "text": "A map of the island.", "reading_order": 2},
        ],
        "term_mentions": [{"term": "island", "block_id": "scan-body"}],
        "illustrations": ["A map of the island."],
    }
)


class TestExtractPage:
    @pytest.mark.asyncio
    async def test_parses_typed_blocks_term_mentions_and_legacy_text(self):
        """A valid Pi response becomes a typed PageExtraction in reading order."""
        from btran.source_extractor import extract_page, legacy_page_text

        proc = _make_mock_proc(stdout=_VALID_OUTPUT)
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            extraction = await extract_page(
                image_path=Path("page-001.png"),
                source_lang="en",
                model="vision-model",
                sha256="a" * 64,
                phash="b" * 16,
                page_number=1,
            )

        assert isinstance(extraction, PageExtraction)
        assert extraction.blocks == [
            SourceBlock("page_1_block_0", "heading", "Chapter One", 0),
            SourceBlock("page_1_block_1", "paragraph", "The first paragraph.", 1),
            SourceBlock("page_1_block_2", "illustration", "A map of the island.", 2),
        ]
        assert extraction.term_mentions == [
            TermMention(term="island", block_id="page_1_block_1")
        ]
        assert extraction.illustrations == ["A map of the island."]
        assert legacy_page_text(extraction) == "Chapter One\n\nThe first paragraph."
        assert extraction.page_number == 1
        assert extraction.sha256 == "a" * 64
        assert extraction.phash == "b" * 16
        assert extraction.source_lang == "en"
        assert extraction.model == "vision-model"

    @pytest.mark.asyncio
    async def test_block_ids_are_deterministic_not_model_supplied(self):
        """The same page and reading order retain canonical IDs despite Pi IDs changing."""
        from btran.source_extractor import extract_page

        first = json.dumps({
            "blocks": [{"id": "arbitrary-a", "type": "paragraph", "text": "One", "reading_order": 7}],
            "term_mentions": [], "illustrations": [],
        })
        second = json.dumps({
            "blocks": [{"id": "arbitrary-b", "type": "paragraph", "text": "One", "reading_order": 7}],
            "term_mentions": [], "illustrations": [],
        })
        procs = [_make_mock_proc(stdout=first), _make_mock_proc(stdout=second)]
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(side_effect=procs),
        ):
            kwargs = dict(
                image_path=Path("page.png"), source_lang="en", model="model",
                sha256="a" * 64, phash="b" * 16, page_number=9,
            )
            one = await extract_page(**kwargs)
            two = await extract_page(**kwargs)

        assert one.blocks[0].id == "page_9_block_7"
        assert two.blocks[0].id == one.blocks[0].id

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "output, error",
        [
            ({"blocks": [{"id": "a", "type": "unknown", "text": "x", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "type"),
            ({"blocks": [{"id": "a", "type": "paragraph", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "text"),
            ({"blocks": [{"id": "a", "type": "paragraph", "text": "x", "reading_order": 0}, {"id": "b", "type": "heading", "text": "y", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "reading_order"),
            ({"blocks": [{"type": "paragraph", "text": "x", "reading_order": 0}], "term_mentions": [], "illustrations": []}, "id"),
        ],
    )
    async def test_rejects_invalid_source_blocks(self, output, error):
        """Every block must carry a source ID, supported type, text, and unique order."""
        from btran.source_extractor import ExtractionError, extract_page

        proc = _make_mock_proc(stdout=json.dumps(output))
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=proc),
        ):
            with pytest.raises(ExtractionError, match=error):
                await extract_page(
                    Path("page.png"), "en", "model", "a" * 64, "b" * 16, 1
                )

    @pytest.mark.asyncio
    async def test_rejects_term_mention_for_unknown_block(self):
        """Term mentions must refer to a source block from the same response."""
        from btran.source_extractor import ExtractionError, extract_page

        output = json.dumps({
            "blocks": [{"id": "known", "type": "paragraph", "text": "text", "reading_order": 0}],
            "term_mentions": [{"term": "term", "block_id": "missing"}],
            "illustrations": [],
        })
        with patch(
            "btran.source_extractor.asyncio.create_subprocess_exec",
            AsyncMock(return_value=_make_mock_proc(stdout=output)),
        ):
            with pytest.raises(ExtractionError, match="block_id"):
                await extract_page(Path("page.png"), "en", "model", "a" * 64, "b" * 16, 1)

    @pytest.mark.asyncio
    async def test_pi_command_is_one_bounded_vision_call(self):
        """Extraction makes one --no-session Pi call with its image attachment and timeout."""
        from btran.source_extractor import EXTRACTION_PROMPT, extract_page

        exec_mock = AsyncMock(return_value=_make_mock_proc(stdout=_VALID_OUTPUT))
        with patch("btran.source_extractor.asyncio.create_subprocess_exec", exec_mock):
            await extract_page(
                Path("/photos/p1.png"), "ja", "gemini-vision", "a" * 64, "b" * 16, 1,
                pi_bin="/bin/pi", timeout=3,
            )

        args, kwargs = exec_mock.call_args
        assert args[:2] == ("/bin/pi", "-p")
        assert args[args.index("--model") + 1] == "gemini-vision"
        assert "--no-session" in args
        assert args[-1].endswith("@/photos/p1.png")
        assert "ja" in args[-1]
        assert "term_mentions" in EXTRACTION_PROMPT
        assert kwargs["stdout"] is asyncio.subprocess.PIPE


class TestExtractionArtifacts:
    def test_to_file_writes_extraction_atomically(self, tmp_path):
        """An existing artifact is atomically replaced by valid extraction JSON."""
        from btran.source_extractor import to_file

        extraction = PageExtraction(
            page_number=2, image_path="p2.png", sha256="a" * 64, phash="b" * 16,
            source_lang="en", model="model",
            blocks=[SourceBlock("page_2_block_0", "paragraph", "Text", 0)],
        )
        path = tmp_path / "nested" / "page-2.json"
        path.parent.mkdir()
        path.write_text("old artifact")

        to_file(extraction, path)

        assert PageExtraction.from_file(path) == extraction
        assert not list(path.parent.glob(".page-2.json.*.tmp"))

    def test_extraction_cache_identity_is_namespaced_and_semantic(self):
        """Extraction keys cannot collide with translations and change with extraction inputs."""
        from btran.source_extractor import extraction_cache_identity

        first = extraction_cache_identity("a" * 64, "en", "vision-a")
        assert first == extraction_cache_identity("a" * 64, "en", "vision-a")
        assert first.startswith("extraction:")
        assert first != extraction_cache_identity("a" * 64, "fr", "vision-a")
        assert first != extraction_cache_identity("a" * 64, "en", "vision-b")

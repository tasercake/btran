"""Regression tests for the typed Wave 2 orchestrator boundary."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from btran.config import Config
from btran.orchestrator import _atomic_write, orchestrator_run, run
from btran.schema import PageExtraction, SourceBlock, TranslatedBlock


def _config(tmp_path: Path) -> Config:
    return Config(input_dir=tmp_path / "in", output_epub=tmp_path / "out.epub", intermediate_dir=tmp_path / "work", source_lang="ja", target_lang="en", max_retries=1)


def _image(path: Path) -> None:
    Image.new("RGB", (600, 600), (1, 2, 3)).save(path)


def _source(path: Path) -> PageExtraction:
    return PageExtraction(1, str(path), "a" * 64, "b" * 16, "ja", "vision", blocks=[SourceBlock("page_1_block_0", "paragraph", "source", 0)])


def test_atomic_write_replaces_complete_checkpoint_without_temp_residue(tmp_path: Path):
    path = tmp_path / "checkpoint.json"
    _atomic_write(path, "first")
    _atomic_write(path, "second")
    assert path.read_text() == "second"
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.asyncio
async def test_orchestrator_run_preserves_gate_one_callable_and_page_order(tmp_path: Path):
    cfg = _config(tmp_path); cfg.input_dir.mkdir(); _image(cfg.input_dir / "z.png"); _image(cfg.input_dir / "a.png")
    sources = [_source(cfg.input_dir / "a.png"), PageExtraction(2, str(cfg.input_dir / "z.png"), "c" * 64, "d" * 16, "ja", "vision", blocks=[SourceBlock("page_2_block_0", "paragraph", "source", 0)])]
    async def translate(source, glossary, **kwargs):
        return [TranslatedBlock(block.id, "translation") for block in source.blocks]
    with patch("btran.orchestrator.extract_page", AsyncMock(side_effect=sources)), patch("btran.orchestrator.translate_blocks", side_effect=translate), patch("btran.orchestrator.build_epub"):
        result = await orchestrator_run(cfg)
    assert result.errors == []
    assert (cfg.intermediate_dir / "page_0001.json").is_file()
    assert (cfg.intermediate_dir / "page_0002.json").is_file()


@pytest.mark.asyncio
async def test_all_page_gate_preserves_existing_output_when_validation_fails(tmp_path: Path):
    cfg = _config(tmp_path); cfg.input_dir.mkdir(); _image(cfg.input_dir / "a.png"); cfg.output_epub.write_bytes(b"old")
    source = _source(cfg.input_dir / "a.png")
    with patch("btran.orchestrator.extract_page", AsyncMock(return_value=source)), patch("btran.orchestrator.translate_blocks", AsyncMock(return_value=[])), patch("btran.orchestrator.build_epub") as epub:
        result = await run(cfg)
    assert result.errors and "page 1" in result.errors[0]
    assert cfg.output_epub.read_bytes() == b"old"
    epub.assert_not_called()

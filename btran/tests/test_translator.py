"""Tests for btran.translator — pi subprocess image translation."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.schema import PageResult


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_mock_proc(stdout="", stderr="", returncode=0):
    """Build an AsyncMock that quacks like an asyncio subprocess."""
    proc = AsyncMock()
    proc.communicate = AsyncMock(return_value=(stdout.encode(), stderr.encode()))
    proc.returncode = returncode
    return proc


_VALID_JSON = json.dumps(
    {
        "page_text": "Hello world",
        "translated_text": "Bonjour le monde",
        "image_descriptions": ["A globe illustration"],
    }
)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


class TestTranslateImage:
    @pytest.mark.asyncio
    async def test_valid_json_returns_pageresult(self):
        """Subprocess returns well-formed JSON → PageResult with all metadata."""
        from btran.translator import translate_image

        mock = _make_mock_proc(stdout=_VALID_JSON)

        with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=mock)):
            result = await translate_image(
                image_path=Path("test.jpg"),
                source_lang="en",
                target_lang="fr",
                model="test-model",
                sha256="abc123",
                phash="def456",
                page_number=1,
            )

        assert isinstance(result, PageResult)
        assert result.page_text == "Hello world"
        assert result.translated_text == "Bonjour le monde"
        assert result.image_descriptions == ["A globe illustration"]
        assert result.page_number == 1
        assert result.sha256 == "abc123"
        assert result.phash == "def456"
        assert result.model == "test-model"
        assert result.source_lang == "en"
        assert result.target_lang == "fr"

    @pytest.mark.asyncio
    async def test_invalid_json_raises_translation_error(self):
        """Non-JSON stdout → TranslationError mentioning parse failure."""
        from btran.translator import TranslationError, translate_image

        mock = _make_mock_proc(stdout="definitely not json {{{")

        with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=mock)):
            with pytest.raises(TranslationError, match="parse|JSON"):
                await translate_image(
                    image_path=Path("test.jpg"),
                    source_lang="en",
                    target_lang="fr",
                    model="m",
                    sha256="a",
                    phash="b",
                    page_number=1,
                )

    @pytest.mark.asyncio
    async def test_subprocess_timeout_raises_translation_error(self):
        """Translation timeout → TranslationError."""
        from btran.translator import TranslationError, translate_image

        async def _hang():
            await asyncio.sleep(10)

        mock = AsyncMock()
        mock.communicate = _hang

        with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=mock)):
            with pytest.raises(TranslationError, match="timed out"):
                await translate_image(
                    image_path=Path("test.jpg"),
                    source_lang="en",
                    target_lang="fr",
                    model="m",
                    sha256="a",
                    phash="b",
                    page_number=1,
                    timeout=0.01,
                )

    @pytest.mark.asyncio
    async def test_nonzero_exit_raises_translation_error(self):
        """Non-zero return code → TranslationError."""
        from btran.translator import TranslationError, translate_image

        mock = _make_mock_proc(stdout="some output", stderr="fatal error", returncode=1)

        with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=mock)):
            with pytest.raises(TranslationError, match="exit|non-zero"):
                await translate_image(
                    image_path=Path("test.jpg"),
                    source_lang="en",
                    target_lang="fr",
                    model="m",
                    sha256="a",
                    phash="b",
                    page_number=1,
                )

    @pytest.mark.asyncio
    async def test_missing_fields_raises_translation_error(self):
        """JSON missing required fields → TranslationError."""
        from btran.translator import TranslationError, translate_image

        incomplete = json.dumps({"page_text": "only one field"})
        mock = _make_mock_proc(stdout=incomplete)

        with patch("btran.translator.asyncio.create_subprocess_exec", AsyncMock(return_value=mock)):
            with pytest.raises(TranslationError, match="field|missing"):
                await translate_image(
                    image_path=Path("test.jpg"),
                    source_lang="en",
                    target_lang="fr",
                    model="m",
                    sha256="a",
                    phash="b",
                    page_number=1,
                )

    def test_prompt_formatting(self):
        """TRANSLATION_PROMPT format() injects both language codes & expected keys."""
        from btran.translator import TRANSLATION_PROMPT

        prompt = TRANSLATION_PROMPT.format(source_lang="ja", target_lang="en")
        assert "ja" in prompt
        assert "en" in prompt
        assert "page_text" in prompt
        assert "translated_text" in prompt
        assert "image_descriptions" in prompt
        assert "Translate this book page" in prompt

    @pytest.mark.asyncio
    async def test_pi_command_args(self):
        """create_subprocess_exec receives correctly constructed args."""
        from btran.translator import translate_image

        mock = _make_mock_proc(stdout=_VALID_JSON)
        exec_mock = AsyncMock(return_value=mock)

        with patch("btran.translator.asyncio.create_subprocess_exec", exec_mock):
            await translate_image(
                image_path=Path("/home/user/photos/page_01.jpg"),
                source_lang="ja",
                target_lang="en",
                model="gemini-2.5-flash",
                sha256="a",
                phash="b",
                page_number=1,
                pi_bin="/custom/pi",
            )

        # Positional args passed to create_subprocess_exec
        call_args = exec_mock.call_args
        pos_args = call_args[0]
        assert pos_args[0] == "/custom/pi"
        assert "-p" in pos_args
        assert "--model" in pos_args
        idx = pos_args.index("--model")
        assert pos_args[idx + 1] == "gemini-2.5-flash"
        assert "--no-session" in pos_args
        # Last positional arg is the combined prompt + @image_path
        prompt_arg = pos_args[-1]
        assert "@/home/user/photos/page_01.jpg" in prompt_arg
        assert "ja" in prompt_arg
        assert "en" in prompt_arg

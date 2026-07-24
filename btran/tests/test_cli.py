"""Tests for btran.cli."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.cli import main
from btran.config import Config



class TestHelpAndArgs:
    def test_help_displays_usage(self, capsys):
        """--help displays usage text and exits with code 0."""
        with patch.object(sys, "argv", ["btran", "--help"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "usage:" in output

    def test_missing_required_args_exits(self):
        """Missing required positional args raises SystemExit with non-zero."""
        with patch.object(sys, "argv", ["btran"]):
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code != 0


class TestValidRun:
    def test_valid_args_calls_orchestrator(self, capsys):
        """Valid config calls orchestrator.run and prints EPUB path."""
        config = Config(
            input_dir=Path("/tmp"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/btran_test_work"),
            pi_bin="pi",
        )
        mock_run = AsyncMock()

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with patch("btran.cli.orchestrator_run", new=mock_run):
                    main()

        mock_run.assert_awaited_once_with(config)


class TestValidation:
    def test_pi_bin_not_found_exits(self, capsys):
        """When pi_bin doesn't exist, print error and exit non-zero."""
        config = Config(
            input_dir=Path("/tmp"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/btran_test_work"),
            pi_bin="/nonexistent/path/to/pi",
        )

        with patch("btran.cli.load_config", return_value=config):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "not found" in captured.err or "not found" in captured.out

    def test_input_dir_not_found_exits(self, capsys):
        """When input_dir doesn't exist, print error and exit non-zero."""
        config = Config(
            input_dir=Path("/nonexistent/input/dir"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/work"),
            pi_bin="/usr/bin/pi",
        )

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with pytest.raises(SystemExit) as exc:
                    main()
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "does not exist" in captured.err or "does not exist" in captured.out


class TestKeyboardInterrupt:
    def test_keyboard_interrupt_graceful_exit(self, capsys):
        """KeyboardInterrupt during orchestrator.run prints message and exits."""
        config = Config(
            input_dir=Path("/tmp"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/btran_test_work"),
            pi_bin="pi",
        )
        mock_run = AsyncMock(side_effect=KeyboardInterrupt)

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with patch("btran.cli.orchestrator_run", new=mock_run):
                    with pytest.raises(SystemExit) as exc:
                        main()
        assert exc.value.code != 0
        captured = capsys.readouterr()
        assert "Interrupted" in captured.out or "Interrupted" in captured.err

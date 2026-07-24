"""Tests for btran.cli."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from btran.cli import main
from btran.config import Config
from btran.orchestrator import RunResult


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
        mock_run = AsyncMock(return_value=RunResult(errors=[]))

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with patch("btran.cli.orchestrator_run", new=mock_run):
                    main()

        mock_run.assert_awaited_once()
        # on_page_error callback should be passed
        call_args, call_kwargs = mock_run.call_args
        assert "on_page_error" in call_kwargs


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


class TestFailureExit:
    def test_errors_in_run_result_exit_nonzero_and_print_summary(self, capsys):
        """When RunResult has errors, exit non-zero and print failure summary."""
        config = Config(
            input_dir=Path("/tmp"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/btran_test_work"),
            pi_bin="pi",
        )
        mock_run = AsyncMock(
            return_value=RunResult(errors=[
                "[btran] page 2 failed: exhausted retries",
                "[btran] page 5 failed: JSON parse error",
            ])
        )

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with patch("btran.cli.orchestrator_run", new=mock_run):
                    with pytest.raises(SystemExit) as exc:
                        main()

        assert exc.value.code == 1
        captured = capsys.readouterr()
        assert "2 page(s) failed" in captured.err
        assert "page 2" not in captured.err
        assert "page 5" not in captured.err

    def test_no_errors_exits_zero(self):
        """When RunResult has no errors, exit code is 0."""
        config = Config(
            input_dir=Path("/tmp"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/btran_test_work"),
            pi_bin="pi",
        )
        mock_run = AsyncMock(return_value=RunResult(errors=[]))

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with patch("btran.cli.orchestrator_run", new=mock_run):
                    main()  # Should not raise SystemExit

    def test_on_page_error_streams_to_stderr(self, capsys):
        """CLI's on_page_error callback writes page failures to stderr during run."""
        config = Config(
            input_dir=Path("/tmp"),
            output_epub=Path("/tmp/out.epub"),
            source_lang="en",
            target_lang="es",
            model="gpt-4o",
            intermediate_dir=Path("/tmp/btran_test_work"),
            pi_bin="pi",
        )

        # Simulate orchestrator calling the on_page_error callback
        captured_errors: list[tuple[int, str]] = []

        async def fake_run(config_arg, on_page_error=None):
            if on_page_error:
                on_page_error(3, "test failure on page 3")
                on_page_error(7, "test failure on page 7")
            return RunResult(errors=[
                "[btran] page 3 failed: test failure on page 3",
                "[btran] page 7 failed: test failure on page 7",
            ])

        with patch("btran.cli.load_config", return_value=config):
            with patch("btran.cli.shutil.which", return_value="/usr/bin/pi"):
                with patch("btran.cli.orchestrator_run", side_effect=fake_run):
                    with pytest.raises(SystemExit) as exc:
                        main()

        captured = capsys.readouterr()
        # Streaming errors should appear in stderr
        assert "page 3" in captured.err
        assert "test failure on page 3" in captured.err
        assert "page 7" in captured.err

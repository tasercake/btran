"""Regression tests for identity-safe direct process cleanup."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import btran.process_cleanup as cleanup


@pytest.mark.parametrize("cause", [None, "failure", object()])
def test_popen_rejects_invalid_cause_before_process_operations(cause):
    proc = Mock()
    with patch.object(cleanup, "_capture") as capture, pytest.raises(ValueError):
        cleanup.cleanup_popen(proc, cause=cause)
    capture.assert_not_called()


@pytest.mark.parametrize("cause", [None, "cancellation", object()])
@pytest.mark.asyncio
async def test_async_cleanup_rejects_invalid_cause_before_process_operations(cause):
    proc = Mock()
    with patch.object(cleanup, "_capture") as capture, pytest.raises(ValueError):
        await cleanup.cleanup_async_process(proc, cause=cause)
    capture.assert_not_called()


def test_cleanup_requires_cause():
    proc = Mock()
    with pytest.raises(TypeError):
        cleanup.cleanup_popen(proc)
    with pytest.raises(TypeError):
        cleanup.cleanup_async_process(proc)


@pytest.mark.parametrize(
    ("sig", "method"),
    [(cleanup.signal.SIGTERM, "terminate"), (cleanup.signal.SIGKILL, "kill")],
)
def test_direct_fallback_does_not_signal_reused_pid(sig, method):
    """A failed group signal must not turn a captured PID into a reused-PID kill."""
    captured = cleanup._ProcessRef(4242, 10)
    proc = SimpleNamespace(pid=4242, returncode=None, poll=Mock(return_value=None),
                           terminate=Mock(), kill=Mock())
    tracked = cleanup._TrackedProcesses(leader=captured, pipe_ids=set(), refs={4242: captured})

    with patch.object(cleanup, "_signal_group", return_value=False), \
         patch.object(cleanup, "_proc_ref", return_value=cleanup._ProcessRef(4242, 11)), \
         patch.object(cleanup.os, "kill") as kill:
        cleanup._signal_phase(proc, tracked, sig)

    kill.assert_not_called()
    getattr(proc, method).assert_not_called()


def test_direct_fallback_does_not_signal_child_already_reaped_by_poll():
    """poll() exit race also suppresses raw proc.terminate()."""
    captured = cleanup._ProcessRef(4242, 10)
    proc = SimpleNamespace(pid=4242, returncode=None, poll=Mock(return_value=0),
                           terminate=Mock(), kill=Mock())
    tracked = cleanup._TrackedProcesses(leader=captured, pipe_ids=set(), refs={})

    with patch.object(cleanup, "_signal_group", return_value=False), \
         patch.object(cleanup, "_proc_ref", return_value=captured):
        cleanup._signal_phase(proc, tracked, cleanup.signal.SIGTERM)

    proc.terminate.assert_not_called()


def test_direct_fallback_signals_only_matching_live_captured_child():
    captured = cleanup._ProcessRef(4242, 10)
    proc = SimpleNamespace(pid=4242, returncode=None, poll=Mock(return_value=None),
                           terminate=Mock(), kill=Mock())
    tracked = cleanup._TrackedProcesses(leader=captured, pipe_ids=set(), refs={})

    with patch.object(cleanup, "_signal_group", return_value=False), \
         patch.object(cleanup, "_proc_ref", return_value=captured):
        cleanup._signal_phase(proc, tracked, cleanup.signal.SIGTERM)

    proc.terminate.assert_called_once_with()

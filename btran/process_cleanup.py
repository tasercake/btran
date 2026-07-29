"""Bounded, identity-safe cleanup for isolated external subprocesses."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import os
import signal
import subprocess
import time
from typing import Any, Iterable

TERM_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 2.0
TAIL_LIMIT = 8_192


@dataclass(frozen=True)
class _ProcessRef:
    pid: int
    start_time: int


@dataclass
class _TrackedProcesses:
    leader: _ProcessRef | None
    pipe_ids: set[str]
    refs: dict[int, _ProcessRef] = field(default_factory=dict)

    def merge(self, other: "_TrackedProcesses") -> None:
        self.refs.update(other.refs)
        self.pipe_ids.update(other.pipe_ids)


def tail(value: str | bytes | None, limit: int = TAIL_LIMIT) -> str:
    """Decode and cap diagnostic output without exposing unbounded tails."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value if len(value) <= limit else value[-limit:] + "…[truncated]"


def _proc_ref(pid: int) -> _ProcessRef | None:
    """Return Linux PID identity; start time prevents PID-reuse signaling."""
    if pid <= 0 or os.name != "posix" or not os.path.isdir("/proc"):
        return None
    try:
        stat = open(f"/proc/{pid}/stat", encoding="ascii").read()
        rest = stat.rsplit(")", 1)[1].split()
        # /proc stat field 22; ``rest`` begins at field 3 (state).
        return _ProcessRef(pid, int(rest[19]))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, IndexError):
        return None


def _proc_table() -> dict[int, tuple[_ProcessRef, int]]:
    table: dict[int, tuple[_ProcessRef, int]] = {}
    if os.name != "posix" or not os.path.isdir("/proc"):
        return table
    try:
        entries = os.listdir("/proc")
    except OSError:
        return table
    for entry in entries:
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            stat = open(f"/proc/{pid}/stat", encoding="ascii").read()
            rest = stat.rsplit(")", 1)[1].split()
            table[pid] = (_ProcessRef(pid, int(rest[19])), int(rest[1]))
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError, ValueError, IndexError):
            continue
    return table


def _pipe_ids_from_fds(fds: Iterable[int]) -> set[str]:
    result: set[str] = set()
    for fd in fds:
        try:
            target = os.readlink(f"/proc/self/fd/{fd}")
        except (OSError, ValueError):
            continue
        if target.startswith("pipe:[") and target.endswith("]"):
            result.add(target)
    return result


def _holder_refs(pipe_ids: set[str], leader: _ProcessRef | None) -> dict[int, _ProcessRef]:
    """Find private-pipe writers started with this worker, never PID-only refs.

    A process cannot acquire this private pipe unless it inherited or received it
    from the launched worker.  Requiring its start time to be no older than the
    worker additionally avoids signaling an unrelated reused/long-lived PID.
    """
    if not pipe_ids or os.name != "posix" or not os.path.isdir("/proc"):
        return {}
    refs: dict[int, _ProcessRef] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return refs
    for entry in entries:
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        pid = int(entry)
        ref = _proc_ref(pid)
        if ref is None or (leader is not None and ref.start_time < leader.start_time):
            continue
        try:
            if any(os.readlink(f"/proc/{pid}/fd/{fd}") in pipe_ids for fd in os.listdir(f"/proc/{pid}/fd")):
                refs[pid] = ref
        except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
            continue
    return refs


def _descendant_refs(leader: _ProcessRef | None) -> dict[int, _ProcessRef]:
    if leader is None:
        return {}
    table = _proc_table()
    current_leader = table.get(leader.pid)
    if current_leader is None or current_leader[0] != leader:
        return {}
    descendants: dict[int, _ProcessRef] = {}
    for pid, (ref, parent) in table.items():
        seen: set[int] = set()
        while parent not in seen:
            if parent == leader.pid:
                descendants[pid] = ref
                break
            seen.add(parent)
            parent_entry = table.get(parent)
            if parent_entry is None:
                break
            parent = parent_entry[1]
    return descendants


def _capture(pid: int, pipe_ids: set[str]) -> _TrackedProcesses:
    leader = _proc_ref(pid)
    tracked = _TrackedProcesses(leader=leader, pipe_ids=set(pipe_ids))
    tracked.refs.update(_descendant_refs(leader))
    tracked.refs.update(_holder_refs(tracked.pipe_ids, leader))
    if leader is not None:
        tracked.refs[leader.pid] = leader
    return tracked


def _original_group_alive(pid: int, tracked: _TrackedProcesses) -> bool:
    """Confirm PGID still belongs to recorded worker before group signaling."""
    if os.name != "posix":
        return False
    # Without procfs there is no start-time identity available; retain normal
    # process-group cleanup rather than pretending a PID-only check is safe.
    if tracked.leader is None:
        return not os.path.isdir("/proc")
    if _proc_ref(pid) == tracked.leader:
        return True
    # Leader may exit on TERM while its original group remains. A recorded,
    # identity-checked member proves this PGID is still that original group.
    for ref in tracked.refs.values():
        if _proc_ref(ref.pid) != ref:
            continue
        try:
            if os.getpgid(ref.pid) == pid:
                return True
        except (ProcessLookupError, PermissionError, OSError):
            continue
    return False


def _signal_group(pid: int, tracked: _TrackedProcesses, sig: signal.Signals) -> bool:
    if not _original_group_alive(pid, tracked):
        return False
    try:
        os.killpg(pid, sig)
        return True
    except (ProcessLookupError, PermissionError, OSError):
        return False


def _signal_refs(refs: Iterable[_ProcessRef], sig: signal.Signals) -> None:
    for ref in refs:
        # Re-read identity immediately before each signal. Never signal PID reuse.
        if _proc_ref(ref.pid) != ref:
            continue
        try:
            os.kill(ref.pid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            pass


def _direct_child_is_current(proc: Any, tracked: _TrackedProcesses) -> bool:
    """Prove direct-process fallback still targets captured child, not PID reuse."""
    leader = tracked.leader
    try:
        pid = proc.pid
    except (AttributeError, OSError):
        return False
    if leader is None or pid != leader.pid:
        return False
    # Popen has poll(); asyncio Process exposes returncode instead.  A polling
    # race means no signal: cleanup may still signal separately tracked owners.
    poll = getattr(proc, "poll", None)
    if callable(poll):
        try:
            if poll() is not None:
                return False
        except (ProcessLookupError, PermissionError, OSError, ValueError, AttributeError):
            return False
    elif getattr(proc, "returncode", None) is not None:
        return False
    # This is deliberately last: immediately before proc.terminate()/kill().
    return _proc_ref(leader.pid) == leader


def _signal_phase(proc: Any, tracked: _TrackedProcesses, sig: signal.Signals) -> None:
    group_signalled = _signal_group(proc.pid, tracked, sig)
    _signal_refs(tracked.refs.values(), sig)
    # Direct fallback is safe only while the captured direct child identity is
    # still live. proc.terminate()/kill() otherwise risks a reused PID.
    if not group_signalled and _direct_child_is_current(proc, tracked):
        try:
            proc.kill() if sig == signal.SIGKILL else proc.terminate()
        except (ProcessLookupError, PermissionError, OSError, AttributeError):
            pass


def cleanup_popen(proc: subprocess.Popen[Any], *, term_grace: float = TERM_GRACE_SECONDS,
                  kill_grace: float = KILL_GRACE_SECONDS) -> tuple[str, str]:
    """TERM/KILL worker group plus escaped pipe owners; every wait is bounded."""
    streams = (proc.stdout, proc.stderr)
    tracked = _capture(proc.pid, _pipe_ids_from_fds(
        stream.fileno() for stream in streams if stream is not None
    ))
    _signal_phase(proc, tracked, signal.SIGTERM)
    term_deadline = time.monotonic() + term_grace
    first_out, first_err = _communicate_popen(proc, term_deadline)
    # Leader may have exited after TERM, so refresh pipe owners before KILL.
    tracked.merge(_capture(proc.pid, tracked.pipe_ids))
    _signal_phase(proc, tracked, signal.SIGKILL)
    kill_deadline = time.monotonic() + kill_grace
    killed_out, killed_err = _communicate_popen(proc, kill_deadline)
    try:
        proc.wait(timeout=max(0.0, kill_deadline - time.monotonic()))
    except (subprocess.TimeoutExpired, OSError):
        pass
    return killed_out or first_out, killed_err or first_err


def _communicate_popen(proc: subprocess.Popen[Any], deadline: float) -> tuple[str, str]:
    try:
        out, err = proc.communicate(timeout=max(0.0, deadline - time.monotonic()))
        return tail(out), tail(err)
    except subprocess.TimeoutExpired as exc:
        return tail(exc.output), tail(exc.stderr)


def _async_pipe_ids(proc: asyncio.subprocess.Process) -> set[str]:
    fds: list[int] = []
    for stream in (getattr(proc, "stdout", None), getattr(proc, "stderr", None)):
        transport = getattr(stream, "_transport", None)
        pipe = transport.get_extra_info("pipe") if transport is not None else None
        try:
            fds.append(pipe.fileno())
        except (AttributeError, OSError, ValueError):
            continue
    return _pipe_ids_from_fds(fds)


async def cleanup_async_process(proc: asyncio.subprocess.Process, *, term_grace: float = TERM_GRACE_SECONDS,
                                kill_grace: float = KILL_GRACE_SECONDS) -> tuple[str, str]:
    """Async counterpart of :func:`cleanup_popen`, with bounded communicate/reap."""
    tracked = _capture(proc.pid, _async_pipe_ids(proc))
    _signal_phase(proc, tracked, signal.SIGTERM)
    first_out, first_err = await _communicate_async(proc, term_grace)
    tracked.merge(_capture(proc.pid, tracked.pipe_ids))
    _signal_phase(proc, tracked, signal.SIGKILL)
    killed_out, killed_err = await _communicate_async(proc, kill_grace)
    return killed_out or first_out, killed_err or first_err


async def _communicate_async(proc: asyncio.subprocess.Process, seconds: float) -> tuple[str, str]:
    try:
        out, err = await asyncio.wait_for(proc.communicate(), timeout=max(0.0, seconds))
        return tail(out), tail(err)
    except (asyncio.TimeoutError, ProcessLookupError):
        return "", ""

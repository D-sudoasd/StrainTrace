"""Cross-platform project lock management and stale-lock recovery."""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import socket
import time
import uuid
from typing import Any

from ._io import (
    AgentError,
    _api_attr,
    _canonical,
    _project_root,
    _read_json,
    _unlink_with_retry,
)


def _read_lock_details(lock: Path) -> dict[str, Any] | None:
    for delay in (0.0, 0.02, 0.05):
        if delay:
            time.sleep(delay)
        try:
            return _read_json(lock)
        except AgentError as exc:
            if exc.code == "PROJECT_FILE_MISSING":
                return None
    return {"corrupt": True}


@contextmanager
def _project_lock(root: Path, *, purpose: str):
    lock = root / ".agent.lock"
    payload = {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "process_start_token": _process_start_token(os.getpid()),
        "purpose": purpose,
        "created_unix_s": time.time(),
        "owner_token": uuid.uuid4().hex,
    }
    payload_bytes = _canonical(payload) + b"\n"
    temp_lock = root / f".agent.lock.{payload['owner_token']}.tmp"
    acquired = False
    try:
        with temp_lock.open("xb") as stream:
            stream.write(payload_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp_lock, lock)
            acquired = True
        except FileExistsError as exc:
            existing = _read_lock_details(lock)
            raise AgentError(
                "PROJECT_LOCKED",
                "another writer owns this project; stale-lock recovery is an explicit operation",
                path=str(lock),
                details=existing,
            ) from exc
        except OSError:
            # Windows rename fails if the target exists, so it publishes the
            # already-fsynced payload atomically even without hardlink support.
            if os.name == "nt":
                try:
                    os.rename(temp_lock, lock)
                    acquired = True
                except FileExistsError as exc:
                    existing = _read_lock_details(lock)
                    raise AgentError(
                        "PROJECT_LOCKED",
                        "another writer owns this project; stale-lock recovery is an explicit operation",
                        path=str(lock), details=existing,
                    ) from exc
            else:
                # POSIX rename may overwrite a lock; use exclusive creation.
                # Some filesystems have no portable atomic no-replace rename.
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                except FileExistsError as exc:
                    existing = _read_lock_details(lock)
                    raise AgentError(
                        "PROJECT_LOCKED",
                        "another writer owns this project; stale-lock recovery is an explicit operation",
                        path=str(lock), details=existing,
                    ) from exc
                try:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(payload_bytes)
                        stream.flush()
                        os.fsync(stream.fileno())
                    acquired = True
                except BaseException:
                    try:
                        _unlink_with_retry(lock)
                    except OSError:
                        pass
                    raise
    finally:
        try:
            _unlink_with_retry(temp_lock)
        except OSError:
            pass
    try:
        yield
    finally:
        if acquired:
            try:
                current = _read_json(lock)
                if current.get("owner_token") == payload["owner_token"]:
                    _unlink_with_retry(lock)
            except (AgentError, FileNotFoundError, PermissionError):
                pass


def _process_start_token(pid: int) -> str | None:
    if os.name != "nt":
        path_cls = _api_attr("Path", Path)
        try:
            raw = path_cls(f"/proc/{pid}/stat").read_text(encoding="utf-8", errors="replace")
            rparen = raw.rfind(")")
            if rparen < 0:
                return None
            # After ") ", index 0 is field 3 (state), so field 22 (starttime) is index 19.
            return raw[rparen + 1:].split()[19]
        except (OSError, IndexError):
            return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetProcessTimes.argtypes = (wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4))
    kernel32.GetProcessTimes.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return None
    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    try:
        if not kernel32.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user)):
            return None
        stamp = (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        return f"{stamp:016x}"
    finally:
        kernel32.CloseHandle(handle)


def _pid_state(pid: int, expected_start_token: str | None) -> bool | None:
    """Return alive/dead/unknown without opening a process for termination."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            return False if error == 87 else None  # ERROR_INVALID_PARAMETER means no such process.
        code = wintypes.DWORD()
        try:
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return None
            if code.value != 259:  # STILL_ACTIVE
                return False
            # With no recorded token, an active PID is conservatively alive.
            # This also prevents recovery of a legacy empty lock held by it.
            if expected_start_token is None:
                return True
            actual_start = _process_start_token(pid)
            if actual_start is None:
                return None
            if actual_start != expected_start_token:
                return False  # PID has been reused by a different process.
            return True
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None
    if expected_start_token is not None:
        actual_start = _process_start_token(pid)
        if actual_start is None:
            return None
        if actual_start != expected_start_token:
            return False  # PID has been reused by a different process.
    return True


def recover_lock(project: str | os.PathLike[str], *, expected_pid: int) -> dict[str, Any]:
    root = _project_root(project)
    lock = root / ".agent.lock"
    if type(expected_pid) is not int or expected_pid <= 0:
        raise AgentError("LOCK_PID_INVALID", "expected_pid must be a positive integer", path="$.expected_pid")
    if lock.is_file():
        try:
            if lock.stat().st_size == 0:
                time.sleep(0.05)
                if lock.is_file() and lock.read_bytes() == b"":
                    state = _pid_state(expected_pid, None)
                    if state is True:
                        raise AgentError("LOCK_OWNER_ALIVE", f"PID {expected_pid} is still alive; lock cannot be recovered", path=str(lock))
                    if state is None:
                        raise AgentError("LOCK_OWNER_UNKNOWN", f"cannot safely determine whether PID {expected_pid} is alive", path=str(lock))
                    if lock.is_file() and lock.read_bytes() == b"":
                        _unlink_with_retry(lock)
                        return {"project": str(root), "recovered_pid": expected_pid, "recovered": True, "empty_lock_recovered": True}
        except FileNotFoundError:
            pass
    record = _read_json(lock)
    if record.get("host") != socket.gethostname():
        raise AgentError("LOCK_REMOTE_OWNER", "lock host differs from this machine; remote ownership cannot be verified", path=str(lock))
    if record.get("pid") != expected_pid:
        raise AgentError("LOCK_PID_MISMATCH", "recorded lock PID does not match expected_pid", path="$.expected_pid", details={"recorded_pid": record.get("pid")})
    state = _pid_state(expected_pid, record.get("process_start_token"))
    if state is True:
        raise AgentError("LOCK_OWNER_ALIVE", f"PID {expected_pid} is still alive; lock cannot be recovered", path=str(lock))
    if state is None:
        raise AgentError("LOCK_OWNER_UNKNOWN", f"cannot safely determine whether PID {expected_pid} is alive", path=str(lock))
    # Re-read immediately before unlink to avoid deleting a replacement lock.
    if _read_json(lock).get("owner_token") != record.get("owner_token"):
        raise AgentError("LOCK_CHANGED", "lock owner changed during recovery", path=str(lock))
    _unlink_with_retry(lock)
    return {"project": str(root), "recovered_pid": expected_pid, "recovered": True}

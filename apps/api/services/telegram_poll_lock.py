"""Ensure only one local Telegram polling process runs at a time."""

from __future__ import annotations

import atexit
import os
import sys
from pathlib import Path

LOCK_PATH = Path(__file__).resolve().parents[3] / ".telegram_poll.lock"


def _pid_running(pid: int) -> bool:
    """Cross-platform check whether a process ID is still alive."""
    if pid <= 0:
        return False

    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return exit_code.value == STILL_ACTIVE
                return False
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False

    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError, SystemError):
        return False
    return True


def acquire_poll_lock() -> None:
    """Exit if another telegram_poll.py instance is already running."""
    if LOCK_PATH.exists():
        try:
            old_pid = int(LOCK_PATH.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            old_pid = 0
        if _pid_running(old_pid):
            print(
                f"ERROR: Ya hay un polling activo (PID {old_pid}).\n"
                "Cierra la otra ventana de start-telegram.bat antes de abrir otra.\n"
                "Si usas n8n con Telegram Trigger, desactívalo o no ejecutes polling local.",
                file=sys.stderr,
            )
            sys.exit(1)
        LOCK_PATH.unlink(missing_ok=True)

    LOCK_PATH.write_text(str(os.getpid()), encoding="utf-8")

    def _release() -> None:
        try:
            if LOCK_PATH.exists() and LOCK_PATH.read_text(encoding="utf-8").strip() == str(os.getpid()):
                LOCK_PATH.unlink(missing_ok=True)
        except OSError:
            pass

    atexit.register(_release)

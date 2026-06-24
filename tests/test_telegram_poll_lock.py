"""Tests for telegram poll lock on Windows and Unix."""

import os
import sys

from apps.api.services import telegram_poll_lock as lock_mod


def test_pid_running_current_process():
    assert lock_mod._pid_running(os.getpid()) is True


def test_pid_running_dead_pid():
    assert lock_mod._pid_running(999999) is False


def test_acquire_poll_lock_releases_stale_lock(tmp_path, monkeypatch):
    stale = tmp_path / ".telegram_poll.lock"
    stale.write_text("999999", encoding="utf-8")
    monkeypatch.setattr(lock_mod, "LOCK_PATH", stale)
    lock_mod.acquire_poll_lock()
    assert stale.read_text(encoding="utf-8") == str(os.getpid())

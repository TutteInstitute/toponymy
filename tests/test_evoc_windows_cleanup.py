"""Interrupted Windows fits must release mapped input and inherited logs."""

import ctypes
from ctypes import wintypes
import json
import os
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(sys.platform != "win32", reason="Windows mapped-file cleanup")
def test_stop_child_releases_initialized_windows_files(tmp_path):
    from toponymy import _evoc

    mapped_path = tmp_path / "vectors.bin"
    log_path = tmp_path / "child.log"
    marker = tmp_path / "initialized.json"
    mapped_path.write_bytes(b"data")
    child_code = """import json, mmap, os, pathlib, sys, time
mapped, marker = map(pathlib.Path, sys.argv[1:])
with mapped.open('r+b') as stream, mmap.mmap(stream.fileno(), 0) as mapping:
    marker.write_text(json.dumps({'pid': os.getpid(), 'ppid': os.getppid()}))
    time.sleep(30)
"""
    # Retain a handle to the actual interpreter before stopping its launcher.
    # Cleanup by retained handle cannot terminate an unrelated recycled PID.
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    process = interpreter_handle = None
    try:
        with log_path.open("wb") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-I",
                    "-S",
                    "-c",
                    child_code,
                    str(mapped_path),
                    str(marker),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            deadline = time.monotonic() + 10
            record = None
            while time.monotonic() < deadline and process.poll() is None:
                try:
                    record = json.loads(marker.read_text())
                    break
                except (FileNotFoundError, json.JSONDecodeError):
                    time.sleep(0.01)
            assert record is not None, "Child did not initialize its mapped file"
            assert record["pid"] != os.getpid()
            # SYNCHRONIZE | PROCESS_TERMINATE; access is scoped to this child.
            interpreter_handle = kernel.OpenProcess(
                0x00100000 | 0x0001, False, record["pid"]
            )
            assert interpreter_handle, ctypes.WinError(ctypes.get_last_error())
            assert kernel.WaitForSingleObject(interpreter_handle, 0) == 258
            _evoc._stop_child(process)
        # Do not poll/wait/sleep for the descendant before either unlink: those
        # delays hid this race in the earlier first-wait cancellation probe.
        mapped_path.unlink()
        log_path.unlink()
        assert process.poll() is not None
    finally:
        # This cleanup follows the assertions and cannot turn a failure green.
        # It also covers a test failure before the production stop operation.
        try:
            if interpreter_handle:
                if kernel.WaitForSingleObject(interpreter_handle, 0) == 258:
                    kernel.TerminateProcess(interpreter_handle, 1)
                try:
                    assert kernel.WaitForSingleObject(interpreter_handle, 5000) == 0
                finally:
                    kernel.CloseHandle(interpreter_handle)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)

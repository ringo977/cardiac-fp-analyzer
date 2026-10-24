"""Shared test configuration.

Exit guard. On Python 3.11 the interpreter aborted *after* the whole suite had passed (CI run #50,
24 Oct 2026, PySide6 6.11): ``Fatal Python error: none_dealloc: deallocating None: bug likely caused by a
refcount error in a C extension … Garbage-collecting`` — a reference-count error in a C extension that
only surfaces while the runtime finalises. Python 3.12 is immune (None is immortal there) and 3.9 / 3.10
never hit that code path. The session's own exit status is kept: once pytest has finished (reports
written, summary printed, plugins unconfigured) the process leaves through ``os._exit`` instead of the
final garbage collection. Nothing of the tests themselves depends on this file.
"""
from __future__ import annotations

import atexit
import os
import sys

_SESSION = {"exitstatus": None}


def pytest_sessionfinish(session, exitstatus):
    _SESSION["exitstatus"] = int(exitstatus)


def _leave_before_final_gc():
    code = _SESSION["exitstatus"]
    if code is None:            # pytest never ran a session (import error, --help, …): normal shutdown
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


atexit.register(_leave_before_final_gc)

"""Diagnostic pytest plugin: which THREAD finalizes Qt-owning objects?"""
import gc
import os
import sys
import threading
import weakref

_MAIN = threading.main_thread()


def _say(msg):
    print(f"\n[DIAG] {msg}", file=sys.stderr, flush=True)


def _gc_cb(phase, info):
    t = threading.current_thread()
    if t is not _MAIN and phase == "start":
        _say(f"GC collection STARTED on non-main thread {t.name!r} info={info}")


gc.callbacks.append(_gc_cb)

if os.environ.get("DIAG_GC_OFF") == "1":
    gc.disable()
    _say("cyclic gc DISABLED for this session")


def pytest_configure(config):
    if os.environ.get("DIAG_TRACE_WIN") != "1":
        return
    from apeGmsh.sections import _builder_gui as bg

    orig = bg.SectionBuilderWindow.__init__
    counter = {"n": 0}

    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        counter["n"] += 1
        n = counter["n"]

        def report(n=n):
            t = threading.current_thread()
            mark = "" if t is _MAIN else "  <<< OFF-MAIN-THREAD Qt FINALIZE"
            _say(f"SectionBuilderWindow #{n} finalized on {t.name!r}{mark}")

        weakref.finalize(self, report)

    bg.SectionBuilderWindow.__init__ = patched
    _say("SectionBuilderWindow finalize tracing armed")

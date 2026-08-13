"""Tests — the gmsh runtime lock (ADR 0080 B6 follow-up).

Gmsh is one process-global C++ runtime, neither thread-safe nor
reentrant. The init refcount alone never guarded the ``gmsh.model.*``
calls *between* acquire and release, so two threads could each hold a
valid session and interleave — a C-level abort with no Python
traceback, which is exactly how it was found (a full Windows suite run
died inside the ADR 0080 B6 properties worker with no ``FAILED`` line).

**Why most of this file runs in subprocesses.** The invariant is a
property of a *process*, and the shared suite process is the wrong
place to measure it: apeGmsh sessions legitimately outlive individual
tests (module-scoped fixtures keep one open), the runtime lock is held
for a session's lifetime, and it is re-entrant — so a same-thread probe
cannot even observe that the main thread is holding it. Every threading
assertion below therefore runs in a clean interpreter, which also makes
the positive control automatic rather than a thing someone once did by
hand.

That ambient-session fact is not an artifact, it is the **documented
consequence** of the design: while a session is open in an interpreter,
a background build does not get gmsh, and the ADR 0080 B6 worker
reports a busy runtime instead of hanging (or crashing) — see
``test_busy_runtime_greys_the_panel_with_a_reason``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


TIMEOUT = 300

#: this checkout's ``src`` — a child interpreter would otherwise resolve
#: ``apeGmsh`` through whatever the editable install points at, which in
#: a git worktree is the *main* checkout, not the code under test.
_SRC = str(Path(__file__).resolve().parent.parent / "src")


def _run(body: str) -> "subprocess.CompletedProcess[str]":
    """Run ``body`` in a clean interpreter (no ambient gmsh session)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_SRC, env["PYTHONPATH"]] if env.get("PYTHONPATH") else [_SRC]
    )
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True, text=True, timeout=TIMEOUT, env=env,
    )


#: the shared 8-thread workload: open a session, do real model work,
#: close it, and report the peak number of threads ever inside gmsh.
_WORKLOAD = """
    import threading
    from apeGmsh import apeGmsh

    occupancy = 0
    peak = 0
    seen = threading.Lock()
    errors = []

    def worker(i):
        global occupancy, peak
        try:
            g = apeGmsh(model_name="t%d" % i, verbose=False)
            g.begin()
            try:
                with seen:
                    occupancy += 1
                    peak = max(peak, occupancy)
                g.model.geometry.add_rectangle(
                    x=0.0, y=0.0, z=0.0, dx=1.0, dy=1.0,
                )
                g.mesh.sizing.set_global_size(0.2)
                g.mesh.generation.generate(dim=2)
                with seen:
                    occupancy -= 1
            finally:
                g.end()
        except BaseException as exc:
            errors.append(repr(exc)[:150])

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(120)
    print("PEAK=%d ERRORS=%d LEFT=%d" % (peak, len(errors), occupancy))
    for e in errors[:3]:
        print("ERR", e)
"""


# ─────────────────────────────────────────────────────────────────────
# the invariant — one thread inside gmsh at a time
# ─────────────────────────────────────────────────────────────────────

def test_concurrent_sessions_never_overlap():
    """Eight threads each open a session, touch the model, and close it.
    An occupancy counter incremented on entry and decremented on exit
    must never be observed above 1."""
    proc = _run(_WORKLOAD)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PEAK=1 ERRORS=0 LEFT=0" in proc.stdout, (
        proc.stdout + proc.stderr
    )


def test_without_the_lock_that_workload_kills_the_interpreter():
    """Positive control. Stub the runtime lock out — changing nothing
    else — and the same workload must NOT come back clean, or this
    file is guarding nothing.

    The exact failure is not pinned down because it is not stable: the
    interpreter usually dies outright (no traceback, no useful exit
    status), and when it survives long enough to report, the occupancy
    counter is above 1. Either outcome proves the guard is load-bearing;
    a clean ``PEAK=1`` would prove it is decorative.
    """
    proc = _run(
        """
        import apeGmsh._session as S

        class _NoOp:
            def acquire(self, *a, **k): return True
            def release(self): pass
        S._GMSH_RUNTIME_LOCK = _NoOp()
        """ + _WORKLOAD
    )
    clean = proc.returncode == 0 and "PEAK=1 ERRORS=0 LEFT=0" in proc.stdout
    assert not clean, (
        "the unguarded workload completed cleanly — the runtime lock is "
        "no longer the thing keeping threads out of gmsh:\n" + proc.stdout
    )


def test_nested_sessions_on_one_thread_do_not_self_block():
    """The lock is re-entrant because sessions nest — a Part opened
    inside a session, a helper that acquires briefly. A plain Lock would
    deadlock on the first nested build, so this asserts against a
    timeout, not against an exception."""
    proc = _run(
        """
        from apeGmsh import apeGmsh

        outer = apeGmsh(model_name="outer", verbose=False)
        outer.begin()
        try:
            inner = apeGmsh(model_name="inner", verbose=False)
            inner.begin()
            inner.end()
        finally:
            outer.end()
        print("NESTED OK")
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "NESTED OK" in proc.stdout


# ─────────────────────────────────────────────────────────────────────
# bounded wait — a busy runtime must not hang a worker
# ─────────────────────────────────────────────────────────────────────

def test_gmsh_runtime_lock_times_out_instead_of_hanging():
    """The deadlock rule: the properties worker's UI thread joins it, so
    the worker may never wait forever."""
    proc = _run(
        """
        import threading, time
        from apeGmsh._session import GmshBusyError, gmsh_runtime_lock

        holding = threading.Event()
        release = threading.Event()

        def holder():
            with gmsh_runtime_lock():
                holding.set()
                release.wait(30)

        t = threading.Thread(target=holder)
        t.start()
        assert holding.wait(10)
        start = time.perf_counter()
        try:
            with gmsh_runtime_lock(timeout=0.25):
                print("ACQUIRED (should not happen)")
        except GmshBusyError as exc:
            elapsed = time.perf_counter() - start
            print("BUSY after %.2fs: %s" % (elapsed, "process-global" in str(exc)))
        release.set()
        t.join(30)
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "BUSY after" in proc.stdout and "True" in proc.stdout
    elapsed = float(proc.stdout.split("BUSY after ")[1].split("s:")[0])
    assert elapsed < 5.0


# ─────────────────────────────────────────────────────────────────────
# the properties worker is out of the lock's way entirely
# ─────────────────────────────────────────────────────────────────────

def test_the_properties_worker_builds_while_another_thread_holds_gmsh():
    """The reason the worker meshes out of process rather than merely
    locking: a session the user left open holds the runtime lock for its
    whole lifetime, so a worker that queued for it would never refresh
    the panel. It must build anyway.

    Both lanes: continuum (meshes in a child process) and fiber (opens
    no session at all).
    """
    proc = _run(
        """
        import threading
        from apeGmsh._session import gmsh_runtime_lock
        from apeGmsh.sections import SectionDocument, _properties

        cont = SectionDocument.new(name="t", kind="continuum")
        cont.set_material("s", E=200e3, nu=0.3)
        cont.add_shape("rect_face", id="r", b=4.0, h=4.0, material="s")
        cont.set_mesh(lc=1.0)

        fib = SectionDocument.new(name="f", kind="fiber")
        fib.set_material("m", uniaxial=("ElasticMaterial", {"E": 1.0}))
        fib.add_point(material="m", y=1.0, z=0.0, area=1.0)

        holding = threading.Event()
        release = threading.Event()

        def holder():
            with gmsh_runtime_lock():      # stands in for an open session
                holding.set()
                release.wait(60)

        t = threading.Thread(target=holder)
        t.start()
        assert holding.wait(10)
        a = _properties.build_document(cont.to_dict())
        b = _properties.build_document(fib.to_dict())
        release.set()
        t.join(60)
        print("CONT error=%r area=%s" % (
            a.error, None if a.analysis is None
            else round(a.analysis.geometric().area, 6)))
        print("FIB error=%r identities=%s" % (
            b.error, b.identities is not None))
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "CONT error=None area=16.0" in proc.stdout, proc.stdout
    assert "FIB error=None identities=True" in proc.stdout, proc.stdout


def test_the_properties_worker_never_touches_gmsh_in_process():
    """The invariant that makes the above true: ``build_document`` must
    not acquire the runtime at all. If meshing ever moves back
    in-process, this fails — before the abort does."""
    proc = _run(
        """
        import apeGmsh._session as S
        from apeGmsh.sections import SectionDocument, _properties

        doc = SectionDocument.new(name="t", kind="continuum")
        doc.set_material("s", E=200e3, nu=0.3)
        doc.add_shape("rect_face", id="r", b=4.0, h=4.0, material="s")
        doc.set_mesh(lc=1.0)

        acquires = []
        real = S._gmsh_acquire
        def counting():
            acquires.append(1)
            return real()
        S._gmsh_acquire = counting

        result = _properties.build_document(doc.to_dict())
        print("ERROR=%r ACQUIRES=%d" % (result.error, len(acquires)))
        """
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ERROR=None ACQUIRES=0" in proc.stdout, proc.stdout


# ─────────────────────────────────────────────────────────────────────
# lifecycle guards (no threads — safe in the shared process)
# ─────────────────────────────────────────────────────────────────────

def test_release_without_acquire_still_raises():
    """The underflow guard predates this lock and must not degrade into
    a 'release un-acquired lock' error from a thread that never held
    it."""
    from apeGmsh._session import _gmsh_release

    with pytest.raises(RuntimeError, match="without matching acquire"):
        _gmsh_release()


def test_a_failed_begin_does_not_leak_the_refcount():
    """``begin()`` on an already-open session raises after the acquire
    path; the acquire must not survive it, or the runtime stays locked
    for the process lifetime."""
    import apeGmsh._session as session
    from apeGmsh import apeGmsh

    g = apeGmsh(model_name="dup", verbose=False)
    g.begin()
    try:
        before = session._GMSH_INIT_COUNT
        with pytest.raises(RuntimeError, match="already open"):
            g.begin()
        assert session._GMSH_INIT_COUNT == before
    finally:
        g.end()

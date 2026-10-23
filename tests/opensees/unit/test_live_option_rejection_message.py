"""#2498: the live fork-element gate tells a rejected option from an
unknown element.

A Ladruno build that knows an element class but not one of its options
prints a line such as ``WARNING LadrunoIMKBeam (2D) -- unknown option
'-shear'`` and fails the call. The live emitter's fork-element gate reads
that line (from ``sys.stderr`` and from fd 2) and raises "option X is not
supported by this build", never "requires the Ladruno fork build", which
stays for an unknown class or for output this process cannot read. These
tests pin it with stub engines, without importing openseespy:

* the option is named whether the build raises or silently drops the
  element, and whichever stream it prints on;
* every rejection line the pinned build prints is read: ``unknown option``
  and ``unknown flag``, with the class's ``2d`` / ``3d`` suffix;
* another class's option line, or a class whose name merely contains the
  element's, is not this element's rejection;
* the build's line still reaches the user (the capture echoes it);
* a type verified without an option is still gated on that option;
* a capture that cannot open its temp files closes what it duplicated and
  lets the engine call speak for itself, without chaining its error to the
  ``OSError``.
"""
from __future__ import annotations

import os
import sys
from typing import Any

import pytest

#: The lines ladruno-v20261017 printed on live-fork run 39773542334.
PIN_LINES = {
    "LadrunoIMKBeam -shear": "WARNING LadrunoIMKBeam (2D) -- unknown option '-shear'\n",
    "LadrunoIMKBeam -shearHinge":
        "WARNING LadrunoIMKBeam (2D) -- unknown option '-shearHinge'\n",
    "LadrunoCohesiveHinge -implex":
        "WARNING LadrunoCohesiveHinge tag 1: unknown option '-implex' (ignored)\n",
    "LadrunoForceBeamColumn": "WARNING element type LadrunoForceBeamColumn is unknown\n",
}


# ---------------------------------------------------------------------------
# The live emitter's message: a rejected option is not an unknown element
# ---------------------------------------------------------------------------

_FORK_REQUIRED = "requires the Ladruno fork build"
_UNKNOWN_ELEMENT = "does not know this element"


class _ElementOps:
    """A build whose ``element`` prints ``line`` and then raises, or (with
    ``raises=False``) returns without creating the element."""

    def __init__(self, line: str, *, fd: bool = False, raises: bool = True) -> None:
        self.line, self.fd, self.raises = line, fd, raises

    def element(self, *_args: Any) -> None:
        if self.fd:
            os.write(2, self.line.encode())
        else:
            sys.stderr.write(self.line)
        if self.raises:
            raise RuntimeError("See stderr output")

    def getEleTags(self) -> list[int]:  # noqa: N802 - the OpenSees name
        return []


def _bare_emitter(ops: object) -> Any:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    le = LiveOpsEmitter.__new__(LiveOpsEmitter)
    le._ops = ops
    le._in_partition = False
    le._fork_verified_types = set()
    return le


@pytest.mark.parametrize("fd", [False, True], ids=["sys.stderr", "fd2"])
@pytest.mark.parametrize("raises", [True, False], ids=["raises", "drops"])
def test_a_rejected_option_names_the_option(fd: bool, raises: bool) -> None:
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"], fd=fd,
                                   raises=raises))
    with pytest.raises(RuntimeError) as info:
        le.element("LadrunoIMKBeam", 2, 1, 2, 0.04, 2e11, 1.6e-4, 1, "-shear", 8e10, 0.03)
    msg = str(info.value)
    assert "option -shear is not supported by this build" in msg
    assert "unknown option '-shear'" in msg, "it quotes the build's line"
    assert _FORK_REQUIRED not in msg and _UNKNOWN_ELEMENT not in msg


@pytest.mark.parametrize("raises", [True, False], ids=["raises", "drops"])
def test_an_unknown_class_still_requires_the_fork(raises: bool) -> None:
    unknown = "WARNING element type LadrunoIMKBeam is unknown\n"
    le = _bare_emitter(_ElementOps(unknown, raises=raises))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED) as info:
        le.element("LadrunoIMKBeam", 2, 1, 2)
    assert "is not supported by this build" not in str(info.value)


def test_another_classes_option_line_is_not_this_elements() -> None:
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoCohesiveHinge -implex"]))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
        le.element("LadrunoIMKBeam", 2, 1, 2)
    # A class whose name merely contains the element's is not the element.
    le = _bare_emitter(_ElementOps("WARNING LadrunoIMKBeamX -- unknown option '-q'\n"))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
        le.element("LadrunoIMKBeam", 2, 1, 2)
    # Only a 2d / 3d suffix is the same class; any other is another class.
    for line in ("WARNING LadrunoDispBeamColumn2dX: unknown option '-foo'\n",
                 "WARNING LadrunoDispBeamColumn4d: unknown option '-foo'\n",
                 "ERROR LadrunoUPX 5 -- unknown flag '-foo'\n"):
        etype = "LadrunoUP" if "UPX" in line else "LadrunoDispBeamColumn"
        le = _bare_emitter(_ElementOps(line))
        with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
            le.element(etype, 5, 1, 2, "-foo", 1.0)


#: The rejection lines other fork classes print at FORK_PIN
#: ladruno-v20261023 (bdb69cea5), with ``<x>`` = ``-foo`` and tag 5:
#: ``LadrunoDispBeamColumn2d.cpp:411`` / ``3d.cpp:590``,
#: ``LadrunoForceBeamColumn2d.cpp:580`` / ``3d.cpp:862``,
#: ``OPS_LadrunoUP.cpp:539``.
PIN_REJECTIONS = [
    ("LadrunoDispBeamColumn",
     "WARNING LadrunoDispBeamColumn2d: unknown option '-foo'\n"),
    ("LadrunoDispBeamColumn",
     "WARNING LadrunoDispBeamColumn3d: unknown option '-foo'\n"),
    ("LadrunoForceBeamColumn",
     "WARNING LadrunoForceBeamColumn2d: unknown option '-foo'\n"),
    ("LadrunoForceBeamColumn",
     "WARNING LadrunoForceBeamColumn3d: unknown option '-foo'\n"),
    ("LadrunoUP",
     "ERROR LadrunoUP 5 -- unknown flag '-foo' "
     "(unknown flags are fatal for this element)\n"),
]


@pytest.mark.parametrize(("etype", "line"), PIN_REJECTIONS,
                         ids=[ln.split()[1].rstrip(":") for _t, ln in PIN_REJECTIONS])
def test_every_pin_rejection_line_names_the_option(etype: str, line: str) -> None:
    le = _bare_emitter(_ElementOps(line))
    with pytest.raises(RuntimeError) as info:
        le.element(etype, 5, 1, 2, "-foo", 1.0)
    msg = str(info.value)
    assert f"element {etype}: option -foo is not supported by this build" in msg
    assert line.strip() in msg, "it quotes the build's line"
    assert _FORK_REQUIRED not in msg


def test_the_builds_line_still_reaches_the_user(
    capfd: pytest.CaptureFixture[str],
) -> None:
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"], fd=True))
    with pytest.raises(RuntimeError):
        le.element("LadrunoIMKBeam", 2, 1, 2)
    assert "unknown option '-shear'" in capfd.readouterr().err


def test_unreadable_streams_fall_back_to_the_fork_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from apeGmsh.opensees.emitter import live

    monkeypatch.setattr(live, "_dup_std_fds", lambda _streams: None)
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"]))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED):
        le.element("LadrunoIMKBeam", 2, 1, 2)


class _KnowsClassRejectsShear:
    """A build that knows ``LadrunoIMKBeam`` but rejects its ``-shear``
    option: a call carrying ``-shear`` prints the pin's line and raises; any
    other call creates the element."""

    def __init__(self) -> None:
        self.tags: list[int] = []
        self.probes = 0

    def element(self, ele_type: str, tag: int, *args: Any) -> None:
        if "-shear" in args:
            sys.stderr.write(PIN_LINES["LadrunoIMKBeam -shear"])
            raise RuntimeError("See stderr output")
        self.tags.append(tag)

    def getEleTags(self) -> list[int]:  # noqa: N802 - the OpenSees name
        self.probes += 1
        return list(self.tags)


_IMK = (1, 2, 0.04, 2e11, 1.6e-4, 1)


def test_a_verified_type_is_still_gated_on_a_new_option() -> None:
    """Element 1 without ``-shear`` verifies the type; element 2 with
    ``-shear`` must still name the rejected option, exactly as it does on a
    fresh emitter, not leak the build's raw 'See stderr output' (#2498
    review F1)."""
    le = _bare_emitter(_KnowsClassRejectsShear())
    le.element("LadrunoIMKBeam", 1, *_IMK)
    with pytest.raises(RuntimeError) as after_verified:
        le.element("LadrunoIMKBeam", 2, *_IMK, "-shear", 8e10, 0.03)
    fresh = _bare_emitter(_KnowsClassRejectsShear())
    with pytest.raises(RuntimeError) as on_fresh:
        fresh.element("LadrunoIMKBeam", 2, *_IMK, "-shear", 8e10, 0.03)
    for info in (after_verified, on_fresh):
        assert "option -shear is not supported by this build" in str(info.value)
    assert str(after_verified.value) == str(on_fresh.value)


def test_a_verified_signature_takes_the_fast_path() -> None:
    """The gate probes once per type and option set: repeats skip it, and a
    different tag or option value is the same signature."""
    ops = _KnowsClassRejectsShear()
    le = _bare_emitter(ops)
    le.element("LadrunoIMKBeam", 1, *_IMK)
    le.element("LadrunoIMKBeam", 2, *_IMK)
    le.element("LadrunoIMKBeam", 3, *_IMK, "-mass", 1.0)
    le.element("LadrunoIMKBeam", 4, *_IMK, "-mass", 2.0)
    assert ops.tags == [1, 2, 3, 4]
    assert ops.probes == 2
    assert le._fork_verified_types == {"LadrunoIMKBeam", "LadrunoIMKBeam -mass"}


def test_no_temp_file_runs_uncaptured_and_closes_the_dups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A capture that cannot open its temp files closes the descriptors it
    duplicated and lets the engine call speak for itself: an element the
    build accepts is created, not misreported as needing the fork."""
    from apeGmsh.opensees.emitter import live

    def no_temp_file() -> Any:
        raise OSError("no temp dir")

    duped: list[int] = []
    real_dup = live._dup_std_fds

    def spy_dup(streams: Any) -> Any:
        fds = real_dup(streams)
        assert fds is not None
        duped.extend(fds)
        return fds

    closed: list[int] = []
    real_close = os.close

    def spy_close(fd: int) -> None:
        closed.append(fd)
        real_close(fd)

    monkeypatch.setattr(live.tempfile, "TemporaryFile", no_temp_file)
    monkeypatch.setattr(live, "_dup_std_fds", spy_dup)
    monkeypatch.setattr(live.os, "close", spy_close)
    ops = _KnowsClassRejectsShear()
    _bare_emitter(ops).element("LadrunoIMKBeam", 1, *_IMK)
    assert ops.tags == [1]
    assert len(duped) == 2 and set(duped) <= set(closed)


def test_no_temp_file_does_not_chain_the_engine_error_to_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fallback runs the engine call outside the ``except OSError``, so
    the engine's own error does not carry that OSError as its context."""
    from apeGmsh.opensees.emitter import live

    def no_temp_file() -> Any:
        raise OSError("no temp dir")

    monkeypatch.setattr(live.tempfile, "TemporaryFile", no_temp_file)
    le = _bare_emitter(_ElementOps(PIN_LINES["LadrunoIMKBeam -shear"]))
    with pytest.raises(RuntimeError, match=_FORK_REQUIRED) as info:
        le.element("LadrunoIMKBeam", 2, 1, 2)
    engine_error = info.value.__cause__
    assert isinstance(engine_error, RuntimeError)
    assert str(engine_error) == "See stderr output"
    assert not isinstance(engine_error.__context__, OSError)

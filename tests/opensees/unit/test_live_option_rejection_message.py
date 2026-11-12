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

#2524 adds:

* a build that prints the rejection and still creates the element (Quad,
  CST, LST, Bezier*, Brick* at the pin) is refused by name, not verified;
* a rejected option *value* says "value ... of option ... is not supported";
* the refusal names the build;
* a known option the build says it ignored is refused too;
* the gate keys on the call's shape, so a stray token after a verified call
  is still read;
* a refused element is removed again, and a taken tag or a known class's
  failing call is not called a missing fork;
* ``live`` + ``ladruno_fork`` checks against the real build at FORK_PIN
  (run by the ``live-fork`` lane).
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


#: The rejection lines fork classes print, and then fail the call, at
#: FORK_PIN ladruno-v20261103 (6050d78e1), with ``<x>`` = ``-foo`` and tag
#: 5 (paths under ``SRC/element/``):
#: ``ladrunoDispBeamColumn/LadrunoDispBeamColumn2d.cpp:493`` / ``3d.cpp:754``,
#: ``ladrunoForceBeamColumn/LadrunoForceBeamColumn2d.cpp:722`` /
#: ``3d.cpp:1084``, ``ladrunoUP/OPS_LadrunoUP.cpp:539``,
#: ``ladrunoIMKBeam/OPS_LadrunoIMKBeam.cpp:387``,
#: ``ladrunoMembrane/OPS_LadrunoMembrane.cpp:325``,
#: ``ladrunoKinematicCoupling/OPS_LadrunoKinematicCoupling.cpp:247`` (the
#: other coupling, embedded and rigid-body parsers print the same form).
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
    ("LadrunoIMKBeam", "WARNING LadrunoIMKBeam -- unknown option '-foo'\n"),
    ("LadrunoMembrane",
     "WARNING element LadrunoMembrane 5: unknown option '-foo'\n"),
    ("LadrunoKinematicCoupling",
     "WARNING LadrunoKinematicCoupling: unknown option '-foo'\n"),
]


@pytest.mark.parametrize(("etype", "line"), PIN_REJECTIONS,
                         ids=[f"{t}-{ln.split()[1].rstrip(':')}"
                              for t, ln in PIN_REJECTIONS])
def test_every_pin_rejection_line_names_the_option(etype: str, line: str) -> None:
    le = _bare_emitter(_ElementOps(line))
    with pytest.raises(RuntimeError) as info:
        le.element(etype, 5, 1, 2, "-foo", 1.0)
    msg = str(info.value)
    assert f"element {etype}: option -foo is not supported by this build" in msg
    assert line.strip() in msg, "it quotes the build's line"
    assert "WITHOUT" not in msg, "the build failed the call, it created nothing"
    assert _FORK_REQUIRED not in msg


# ---------------------------------------------------------------------------
# #2524: a build that prints the rejection and still creates the element
# ---------------------------------------------------------------------------

#: The lines the classes that warn and go on print at FORK_PIN
#: ladruno-v20261103 (6050d78e1), with ``<x>`` = ``-bogus`` and tag 5. Each
#: parser's ``else`` branch prints and falls through, so the element is
#: built without the option (paths under ``SRC/element/``):
#: ``ladrunoPlane/OPS_LadrunoQuad.cpp:164``, ``OPS_LadrunoCST.cpp:132``,
#: ``OPS_LadrunoLST.cpp:159`` (unquoted, each its own class name);
#: ``bezierTriangle/OPS_BezierTri6.cpp:187``,
#: ``bezierTetrahedron/OPS_BezierTet10.cpp:226``,
#: ``ladrunoBrick/OPS_LadrunoBrick.cpp:260``, ``OPS_LadrunoBrick20.cpp:163``.
PIN_IGNORED = [
    ("LadrunoQuad", "WARNING LadrunoQuad -- ignoring unknown option -bogus\n"),
    ("LadrunoCST", "WARNING LadrunoCST -- ignoring unknown option -bogus\n"),
    ("LadrunoLST", "WARNING LadrunoLST -- ignoring unknown option -bogus\n"),
    ("BezierTri6", "WARNING unknown option '-bogus' for BezierTri6 5\n"),
    ("BezierTet10", "WARNING unknown option '-bogus' for BezierTet10 5\n"),
    ("LadrunoBrick", "WARNING unknown option '-bogus' for LadrunoBrick 5\n"),
    ("LadrunoBrick20", "WARNING unknown option '-bogus' for LadrunoBrick20 5\n"),
]


class _CreatesAnyway:
    """A build whose ``element`` prints ``line`` (a callable of the call's
    trailing arguments, or a fixed string) and still creates the element
    (``tags`` lists it), stamped ``build`` when one is given. A tag already
    in the domain fails the call the way OpenSees does; ``remove`` takes an
    element out again."""

    def __init__(self, line: Any, *, fd: bool = False,
                 build: "str | None" = None) -> None:
        self.line, self.fd, self.build = line, fd, build
        self.tags: list[int] = []

    def _print(self, text: str) -> None:
        if self.fd:
            os.write(2, text.encode())
        else:
            sys.stderr.write(text)

    def element(self, _ele_type: str, tag: int, *args: Any) -> None:
        if tag in self.tags:
            self._print(f"WARNING could not add element with tag {tag} to the domain\n")
            raise RuntimeError("See stderr output")
        self._print(self.line(args) if callable(self.line) else self.line)
        self.tags.append(tag)

    def remove(self, kind: str, tag: int) -> None:
        assert kind == "element"
        self.tags.remove(tag)

    def getEleTags(self) -> list[int]:  # noqa: N802 - the OpenSees name
        return list(self.tags)

    def ladrunoBuild(self) -> "str | None":  # noqa: N802 - the OpenSees name
        return self.build


class _CreatesAnywayNoEleTags(_CreatesAnyway):
    def getEleTags(self) -> list[int]:  # noqa: N802 - the OpenSees name
        raise AttributeError("getEleTags")


class _CreatesAnywayNoRemove(_CreatesAnyway):
    def remove(self, kind: str, tag: int) -> None:
        raise RuntimeError("no remove")


@pytest.mark.parametrize("fd", [False, True], ids=["sys.stderr", "fd2"])
@pytest.mark.parametrize(("etype", "line"), PIN_IGNORED,
                         ids=[t for t, _ln in PIN_IGNORED])
def test_an_ignored_option_is_refused_though_the_element_was_built(
    etype: str, line: str, fd: bool,
) -> None:
    ops = _CreatesAnyway(line, fd=fd)
    le = _bare_emitter(ops)
    with pytest.raises(RuntimeError) as info:
        le.element(etype, 5, 1, 2, 3, 4, 1, "-bogus")
    msg = str(info.value)
    assert f"element {etype}: option -bogus is not supported by this build" in msg
    assert "WITHOUT" in msg, "it says the element was built without it"
    assert line.strip() in msg, "it quotes the build's line"
    assert _FORK_REQUIRED not in msg
    assert le._fork_verified_types == set(), "a refused signature is not cached"
    assert ops.tags == [], "the element the build made without it is removed"
    assert "removed from the domain again" in msg


def test_an_ignored_option_is_refused_without_get_ele_tags() -> None:
    """A build with no ``getEleTags`` is trusted to have created the
    element, but not to have kept an option it said it ignored. It cannot
    confirm the element, so nothing is removed."""
    ops = _CreatesAnywayNoEleTags(PIN_IGNORED[0][1])
    le = _bare_emitter(ops)
    with pytest.raises(RuntimeError, match="option -bogus is not supported"):
        le.element("LadrunoQuad", 5, 1, 2, 3, 4, 1, "-bogus")
    assert ops.tags == [5]
    quiet = _bare_emitter(_CreatesAnywayNoEleTags(""))
    quiet.element("LadrunoQuad", 5, 1, 2, 3, 4, 1)
    assert quiet._fork_verified_types == {"LadrunoQuad # # # # #", "LadrunoQuad"}


def test_a_clean_success_is_still_verified_once() -> None:
    """No rejection line: created, verified, and the repeat skips the gate."""
    ops = _CreatesAnyway("")
    le = _bare_emitter(ops)
    le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1, "-thick", 0.2)
    le.element("LadrunoQuad", 2, 1, 2, 3, 4, 1, "-thick", 0.3)
    assert ops.tags == [1, 2]
    assert le._fork_verified_types == {"LadrunoQuad # # # # # -thick #",
                                       "LadrunoQuad"}


def test_the_refusal_names_the_build() -> None:
    sha = "6050d78e105bd4fc3a15ddb8ee521aa861a4dafe"
    le = _bare_emitter(_CreatesAnyway(PIN_IGNORED[0][1], build=sha))
    with pytest.raises(RuntimeError, match=f"Ladruno build {sha}"):
        le.element("LadrunoQuad", 5, 1, 2, 3, 4, 1, "-bogus")
    le = _bare_emitter(_CreatesAnyway(PIN_IGNORED[0][1]))
    with pytest.raises(RuntimeError, match="unstamped build"):
        le.element("LadrunoQuad", 5, 1, 2, 3, 4, 1, "-bogus")


def test_another_classes_ignored_line_on_success_is_not_a_refusal() -> None:
    """A material's own 'unknown option ... (ignored)' line (it names the
    material, not the element) does not refuse the element."""
    ops = _CreatesAnyway(PIN_LINES["LadrunoCohesiveHinge -implex"])
    le = _bare_emitter(ops)
    le.element("LadrunoQuad", 5, 1, 2, 3, 4, 1)
    assert ops.tags == [5]


# ---------------------------------------------------------------------------
# Review F1: a stray token after a verified call is still read
# ---------------------------------------------------------------------------


def _quad_parser(args: tuple[Any, ...]) -> str:
    """OPS_LadrunoQuad.cpp at the pin: every token after matTag that is not
    a known option prints 'ignoring unknown option <tok>' (line 164); a
    number read as a string prints as 'Invalid String Input!' (seen on fork
    79e062367 for a stray 0.2)."""
    known = {"-thick": 1, "-geom": 1}
    rest, out, i = list(args[5:]), [], 0
    while i < len(rest):
        tok = rest[i] if isinstance(rest[i], str) else "Invalid String Input!"
        if tok in known:
            i += 1 + known[tok]
            continue
        out.append(f"WARNING LadrunoQuad -- ignoring unknown option {tok}\n")
        i += 1
    return "".join(out)


@pytest.mark.parametrize("stray", [
    (0.2,), ("PlaneStress",), ("-thick", 0.2, "-1e3"), ("-thick", 0.2, "junk"),
], ids=["number", "word", "numeric-flag", "after-a-flag"])
def test_a_stray_token_after_a_verified_call_is_still_refused(
    stray: tuple[Any, ...],
) -> None:
    """A clean call verifies the type, and a later call of the same type with
    a stray token must still be read and refused (before the fix it took the
    fast path, keyed on the ``-alpha`` flags alone)."""
    ops = _CreatesAnyway(_quad_parser)
    le = _bare_emitter(ops)
    le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1)
    le.element("LadrunoQuad", 2, 1, 2, 3, 4, 1, "-thick", 0.2)
    with pytest.raises(RuntimeError, match="is not supported by this build") as info:
        le.element("LadrunoQuad", 3, 1, 2, 3, 4, 1, *stray)
    assert "WITHOUT" in str(info.value)
    assert ops.tags == [1, 2]


def test_a_flag_value_change_is_read_again() -> None:
    """``-geom linear`` verified does not verify ``-geom finite``: a
    string value is part of the call shape."""
    seen: list[tuple[Any, ...]] = []

    def parser(args: tuple[Any, ...]) -> str:
        seen.append(args)
        return ""

    le = _bare_emitter(_CreatesAnyway(parser))
    le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1, "-geom", "linear")
    le.element("LadrunoQuad", 2, 1, 2, 3, 4, 1, "-geom", "linear")
    le.element("LadrunoQuad", 3, 1, 2, 3, 4, 1, "-geom", "finite")
    assert le._fork_verified_types == {
        "LadrunoQuad", "LadrunoQuad # # # # # -geom linear",
        "LadrunoQuad # # # # # -geom finite"}
    assert len(seen) == 3  # the stub saw every call: all three were made


# ---------------------------------------------------------------------------
# Review F2: a refused element is removed; a taken tag is not "needs the fork"
# ---------------------------------------------------------------------------


def test_a_retry_after_a_refusal_builds_on_the_same_tag() -> None:
    ops = _CreatesAnyway(_quad_parser)
    le = _bare_emitter(ops)
    with pytest.raises(RuntimeError, match="option -bogus"):
        le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1, "-bogus")
    assert ops.tags == []
    le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1)
    assert ops.tags == [1]


def test_a_refused_element_that_cannot_be_removed_says_so() -> None:
    ops = _CreatesAnywayNoRemove(_quad_parser)
    le = _bare_emitter(ops)
    with pytest.raises(RuntimeError) as info:
        le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1, "-bogus")
    assert "could NOT be removed" in str(info.value)
    assert ops.tags == [1]


def test_a_taken_tag_is_not_called_a_missing_fork() -> None:
    ops = _CreatesAnywayNoRemove(_quad_parser)
    le = _bare_emitter(ops)
    with pytest.raises(RuntimeError):
        le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1, "-bogus")
    with pytest.raises(RuntimeError) as info:
        le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1)
    msg = str(info.value)
    assert "already in the domain" in msg
    assert _FORK_REQUIRED not in msg


def test_a_known_class_that_fails_another_call_is_not_a_missing_fork() -> None:
    """Once the build has built the class, a later call that fails with no
    line the gate reads says the class is known, not that it needs the
    fork."""

    class _FailsSecond(_CreatesAnyway):
        def element(self, ele_type: str, tag: int, *args: Any) -> None:
            if self.tags:
                raise RuntimeError("See stderr output")
            super().element(ele_type, tag, *args)

    le = _bare_emitter(_FailsSecond(""))
    le.element("LadrunoQuad", 1, 1, 2, 3, 4, 1)
    with pytest.raises(RuntimeError) as info:
        le.element("LadrunoQuad", 2, 1, 2, 3, 4, 1, 0.5)
    assert "knows LadrunoQuad" in str(info.value)
    assert _FORK_REQUIRED not in str(info.value)


# ---------------------------------------------------------------------------
# Review F3: a known option the build ignored is refused
# ---------------------------------------------------------------------------

#: The pin's "ignoring" lines for known options, tag 5 (paths under
#: ``SRC/element/``): ``ladrunoBrick/OPS_LadrunoBrick.cpp:272`` / ``:286`` /
#: ``:293`` / ``:180``, ``ladrunoPlane/OPS_LadrunoQuad.cpp:228`` / ``:219`` /
#: ``:158``, ``OPS_LadrunoCST.cpp:164``, ``OPS_LadrunoLST.cpp:191``,
#: ``bezierTriangle/OPS_BezierTri6.cpp:200``,
#: ``ladrunoKinematicCoupling/OPS_LadrunoKinematicCoupling.cpp:272`` /
#: ``:291``, ``ladrunoEmbeddedNode/LadrunoEmbeddedNode.cpp:241``.
PIN_IGNORED_KNOWN = [
    ("LadrunoBrick", "-damp", "WARNING LadrunoBrick 5: -damp is only supported "
     "with -formulation std|bbar; ignoring the damping object for this "
     "formulation\n"),
    ("LadrunoBrick", "-bulkViscosity", "WARNING LadrunoBrick 5: -bulkViscosity "
     "is only supported with -formulation std|bbar; ignoring it for this "
     "formulation\n"),
    ("LadrunoBrick", "-bulkViscosity", "WARNING LadrunoBrick 5: -bulkViscosity "
     "is only supported with -geom linear (global-frame velocities are "
     "frame-consistent only then); ignoring it\n"),
    ("LadrunoBrick", "-bulkViscosity", "WARNING -bulkViscosity b1 < 0 for "
     "LadrunoBrick 5; ignoring (b1=0)\n"),
    ("LadrunoQuad", "-bulkViscosity", "WARNING LadrunoQuad 5 -- -bulkViscosity "
     "is not applied with -geom finite; ignoring it\n"),
    ("LadrunoQuad", "-bulkViscosity", "WARNING LadrunoQuad 5 -- -bulkViscosity "
     "is only supported with -formulation std|bbar; ignoring it for this "
     "formulation\n"),
    ("LadrunoQuad", "-bulkViscosity", "WARNING LadrunoQuad -- -bulkViscosity "
     "b1 < 0; ignoring (b1=0)\n"),
    ("LadrunoCST", "-bulkViscosity", "WARNING LadrunoCST 5 -- -bulkViscosity "
     "is not applied with -geom finite; ignoring it\n"),
    ("LadrunoLST", "-bulkViscosity", "WARNING LadrunoLST 5 -- -bulkViscosity "
     "is not applied with -geom finite; ignoring it\n"),
    ("BezierTri6", "-bbar", "WARNING BezierTri6 5 - '-bbar' is not valid for "
     "PlaneStress (no volumetric locking); ignoring -bbar\n"),
    ("LadrunoKinematicCoupling", "-alUpdate", "WARNING LadrunoKinematicCoupling "
     "5: -alUpdate has no effect without -enforce al (the penalty formulation "
     "carries no multiplier); ignored\n"),
    ("LadrunoKinematicCoupling", "-bipenalty", "WARNING LadrunoKinematicCoupling: "
     "-bipenalty ignored with -enforce al (augmented Lagrangian needs no mass "
     "penalty)\n"),
    ("LadrunoEmbeddedNode", "-corot", "WARNING LadrunoEmbeddedNode 5: -corot "
     "only applies to a material interface (-matN/-matT*); ignoring "
     "(penalty/isotropic ties are already frame-objective)\n"),
]


@pytest.mark.parametrize(("etype", "opt", "line"), PIN_IGNORED_KNOWN,
                         ids=[f"{t}{o}-{i}" for i, (t, o, _l)
                              in enumerate(PIN_IGNORED_KNOWN)])
def test_a_known_option_the_build_ignored_is_refused(
    etype: str, opt: str, line: str,
) -> None:
    ops = _CreatesAnyway(line)
    le = _bare_emitter(ops)
    with pytest.raises(RuntimeError) as info:
        le.element(etype, 5, 1, 2, 3, 4, 1, opt)
    msg = str(info.value)
    assert f"element {etype}: option {opt} was ignored by this build" in msg
    assert "WITHOUT" in msg and line.strip() in msg
    assert _FORK_REQUIRED not in msg
    assert ops.tags == [] and le._fork_verified_types == set()

# ---------------------------------------------------------------------------
# #2524 scope addition: a rejected option VALUE says so
# ---------------------------------------------------------------------------

#: The lines the pin prints, failing the call, when it knows an option but
#: not its value ``xyz`` (tag 5; paths under ``SRC/element/``):
#: ``ladrunoBrick/OPS_LadrunoBrick.cpp:126`` / ``:141``,
#: ``OPS_LadrunoBrick20.cpp:113``, ``bezierTetrahedron/OPS_BezierTet10.cpp:167``
#: / ``:220``, ``ladrunoPlane/OPS_LadrunoQuad.cpp:107`` / ``:113`` (unquoted),
#: ``ladrunoKinematicCoupling/OPS_LadrunoKinematicCoupling.cpp:197`` /
#: ``:218``, ``ladrunoUP/OPS_LadrunoUP.cpp:390``,
#: ``ladrunoMembrane/OPS_LadrunoMembrane.cpp:198``,
#: ``ladrunoForceBeamColumn/LadrunoForceBeamColumn3d.cpp:554`` / ``2d.cpp:531``.
PIN_VALUES = [
    ("LadrunoBrick", "-formulation", "WARNING unknown -formulation 'xyz' for "
     "LadrunoBrick 5 (use std|bbar|uri|ssp|eas)\n"),
    ("LadrunoBrick", "-hourglass", "WARNING unknown -hourglass 'xyz' for "
     "LadrunoBrick 5 (use viscous|stiffness|physical)\n"),
    ("LadrunoBrick20", "-formulation", "WARNING unknown -formulation 'xyz' for "
     "LadrunoBrick20 5 (use std or uri)\n"),
    ("BezierTet10", "-fbar", "WARNING unknown -fbar 'xyz' for BezierTet10 5 "
     "(use centroid|mean_dilatation)\n"),
    ("BezierTet10", "-geom", "WARNING unknown -geom 'xyz' for BezierTet10 5 "
     "(use linear|corot|finite)\n"),
    ("LadrunoQuad", "-formulation", "WARNING LadrunoQuad -- unknown "
     "-formulation xyz\n"),
    ("LadrunoQuad", "-geom", "WARNING LadrunoQuad -- unknown -geom xyz "
     "(want linear|finite)\n"),
    ("LadrunoKinematicCoupling", "-enforce", "WARNING LadrunoKinematicCoupling: "
     "unknown -enforce 'xyz' (want penalty|al)\n"),
    ("LadrunoKinematicCoupling", "-alUpdate", "WARNING LadrunoKinematicCoupling: "
     "unknown -alUpdate 'xyz' (want ...)\n"),
    ("LadrunoUP", "-formulation", "ERROR LadrunoUP 5 -- unknown -formulation "
     "'xyz' (want std|bbar; ssp/eas/uri are not u-p axes, ADR 71 §3.4)\n"),
    ("LadrunoMembrane", "-method", "WARNING element LadrunoMembrane 5: unknown "
     "-method 'xyz' (expected tributary, oneway, fea_poisson, or fea_plate)\n"),
    ("LadrunoForceBeamColumn", "-Integration", "WARNING "
     "LadrunoForceBeamColumn3d: unknown -Integration type 'xyz'\n"),
    ("LadrunoForceBeamColumn", "-Integration", "WARNING "
     "LadrunoForceBeamColumn2d: invalid -Integration type 'xyz'\n"),
]


@pytest.mark.parametrize(("etype", "opt", "line"), PIN_VALUES,
                         ids=[f"{t}{o}-{i}" for i, (t, o, _l) in enumerate(PIN_VALUES)])
def test_a_rejected_value_names_the_option_and_value(
    etype: str, opt: str, line: str,
) -> None:
    le = _bare_emitter(_ElementOps(line))
    with pytest.raises(RuntimeError) as info:
        le.element(etype, 5, 1, 2, opt, "xyz")
    msg = str(info.value)
    assert (f"element {etype}: value 'xyz' of option {opt} is not supported "
            f"by this build") in msg
    assert line.strip() in msg, "it quotes the build's line"
    assert "WITHOUT" not in msg
    assert _FORK_REQUIRED not in msg


@pytest.mark.parametrize(("etype", "line"), [
    ("LadrunoIMKBeam", "WARNING LadrunoIMKBeam -- invalid -pm P0 (must be > 0)\n"),
    ("LadrunoBrick", "WARNING invalid -bulkViscosity b1 b2 for LadrunoBrick 5\n"),
    ("LadrunoBrick", "WARNING invalid -b body force for LadrunoBrick 5\n"),
])
def test_a_malformed_number_is_not_a_rejected_value(etype: str, line: str) -> None:
    """``invalid -<opt> ...`` without ``type`` is a bad number, not a value
    the build does not know (OPS_LadrunoIMKBeam.cpp:350,
    OPS_LadrunoBrick.cpp:174 / :242): it is not dressed up as one."""
    le = _bare_emitter(_ElementOps(line))
    with pytest.raises(RuntimeError) as info:
        le.element(etype, 5, 1, 2)
    assert "is not supported by this build" not in str(info.value)


# ---------------------------------------------------------------------------
# The real build at FORK_PIN (live-fork lane)
# ---------------------------------------------------------------------------


def _live_emitter() -> Any:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    return LiveOpsEmitter(wipe=True)


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_pin_refuses_a_quad_option_it_ignores() -> None:
    """At the pin LadrunoQuad prints 'ignoring unknown option -bogus' and
    builds the element; the live route refuses by name."""
    em = _live_emitter()
    o = em.ops
    try:
        o.model("basic", "-ndm", 2, "-ndf", 2)
        for n, (x, y) in enumerate([(0, 0), (1, 0), (1, 1), (0, 1)], start=1):
            o.node(n, float(x), float(y))
        o.nDMaterial("ElasticIsotropic", 1, 2.0e11, 0.3)
        with pytest.raises(RuntimeError) as info:
            em.element("LadrunoQuad", 1, 1, 2, 3, 4, 1, "-bogus")
        msg = str(info.value)
        assert "option -bogus is not supported by this build" in msg
        assert "WITHOUT" in msg
        assert "removed from the domain again" in msg
        assert 1 not in list(o.getEleTags() or [])
        # The tag is free again, and a stray thickness after a verified
        # call is still read (review F1/F2).
        em.element("LadrunoQuad", 1, 1, 2, 3, 4, 1)
        # The parser reads the number as a string and prints
        # 'ignoring unknown option Invalid String Input!'.
        with pytest.raises(RuntimeError, match="is not supported by this build"):
            em.element("LadrunoQuad", 2, 1, 2, 3, 4, 1, 0.2)
    finally:
        o.wipe()


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_pin_refuses_a_brick_option_and_a_brick_value() -> None:
    em = _live_emitter()
    o = em.ops
    try:
        o.model("basic", "-ndm", 3, "-ndf", 3)
        corners = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
                   (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        for n, xyz in enumerate(corners, start=1):
            o.node(n, *map(float, xyz))
        o.nDMaterial("ElasticIsotropic", 1, 2.0e11, 0.3)
        with pytest.raises(RuntimeError) as info:
            em.element("LadrunoBrick", 1, *range(1, 9), 1, "-bogus")
        assert "option -bogus is not supported by this build" in str(info.value)
        assert "WITHOUT" in str(info.value)
        with pytest.raises(RuntimeError) as info:
            em.element("LadrunoBrick", 2, *range(1, 9), 1, "-formulation", "xyz")
        assert ("value 'xyz' of option -formulation is not supported"
                in str(info.value))
    finally:
        o.wipe()


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_pin_refuses_a_disp_beam_column_option() -> None:
    """LadrunoDispBeamColumn printed nothing for an unknown flag before fork
    34df942c7 (WP-200); at the pin it prints and fails the call, so the
    live route names the option (no apeGmsh-side options table)."""
    em = _live_emitter()
    o = em.ops
    try:
        o.model("basic", "-ndm", 2, "-ndf", 3)
        o.node(1, 0.0, 0.0)
        o.node(2, 3.0, 0.0)
        o.section("Elastic", 1, 2.0e11, 0.02, 3.0e-4)
        o.geomTransf("Linear", 1)
        o.beamIntegration("Lobatto", 1, 1, 3)
        with pytest.raises(RuntimeError) as info:
            em.element("LadrunoDispBeamColumn", 1, 1, 2, 1, 1, "-bogus")
        msg = str(info.value)
        assert "option -bogus is not supported by this build" in msg
        assert "WITHOUT" not in msg
    finally:
        o.wipe()


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
    """The gate probes once per call shape: repeats skip it, and a
    different tag or numeric value is the same shape."""
    ops = _KnowsClassRejectsShear()
    le = _bare_emitter(ops)
    le.element("LadrunoIMKBeam", 1, *_IMK)
    le.element("LadrunoIMKBeam", 2, *_IMK)
    le.element("LadrunoIMKBeam", 3, *_IMK, "-mass", 1.0)
    le.element("LadrunoIMKBeam", 4, *_IMK, "-mass", 2.0)
    assert ops.tags == [1, 2, 3, 4]
    assert ops.probes == 2
    assert le._fork_verified_types == {
        "LadrunoIMKBeam", "LadrunoIMKBeam # # # # # #",
        "LadrunoIMKBeam # # # # # # -mass #"}


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

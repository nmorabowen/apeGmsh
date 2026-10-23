"""#2457: does each Ladruno solid forward ``materialState`` at ``FORK_PIN``?

#2446 emits a plastic ``ManzariDafalias`` flip on a material two or more
elements carry as::

    setParameter -val 1 -ele <carriers> materialState <tag>
    updateMaterialStage -material <tag> -stage 1

and gates it on :data:`MATERIAL_STATE_ELEMENTS`.  The Ladruno quad / brick
classes were measured to drop the update on the local fork build 79e062367,
so #2446 refused them.  This module runs the same per-element measurement
for every class and every formulation the bridge emits, on the ``live-fork``
lane (a build at ``FORK_PIN``):

* ``reference``: each element its own material tag and its own
  ``updateMaterialStage`` (each runs its own ``Elastic2Plastic()``);
* ``routed``: one tag on both elements, flipped by the two lines above;
* ``first``: one tag, the bare ``updateMaterialStage`` (reaches element 1).

Both elements run the same prescribed homogeneous strain path (every DOF
driven), elastic to t=1, the flip, then on to t=2.  ``routed`` must equal
``reference`` bit for bit and ``first`` must differ from it.

Measured at FORK_PIN 100eb9ae (live-fork run 39874892907): every element
case forwards, bit for bit; on 79e062367 every one drops it.

It also runs the fork's finite-strain wrappers over ManzariDafalias
(``LogStrain``, ``LogStrain2D``, ``InitDefGrad``), refused by #2446 because
over MD the first two ended the process on 79e062367.  They still do at the
pin (``ManzariDafalias::getType`` exits), and ``InitDefGrad`` still refuses
a small-strain inner; the tests pin those outcomes.

Every case runs in a child interpreter, so a fork crash fails one case
instead of ending the lane's pytest process.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.live, pytest.mark.ladruno_fork]

_ROOT = Path(__file__).resolve().parents[3]
_SENTINEL = "MD_STAGE_AT_PIN_RESULT:"

#: Dafalias & Manzari (2004) Toyoura sand, the OpenSees example set
#: (``test_md_stage_all_elements.MD_ARGS``, copied so the child imports
#: nothing from the test tree).
MD_ARGS = [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0, 0.01,
           7.05, 0.968, 1.1, 0.704, 3.5, 4.0, 600.0, 1.42]

_TRI3 = [(0, 0), (1, 0), (0, 1)]
_TRI6 = _TRI3 + [(0.5, 0), (0.5, 0.5), (0, 0.5)]
_QUAD4 = [(0, 0), (1, 0), (1, 1), (0, 1)]
_HEX8 = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
         (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]


def _hex20() -> list[tuple[float, ...]]:
    """A unit hex20 in the fork's serendipity order: Gmsh's etype-17 order
    permuted by the bridge's own ``GMSH_HEX20_TO_SERENDIPITY``."""
    from apeGmsh.opensees._element_capabilities import (
        GMSH_HEX20_TO_SERENDIPITY,
    )

    edges = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (2, 3), (2, 6), (3, 7),
             (4, 5), (4, 7), (5, 6), (6, 7))
    gmsh = [tuple(float(v) for v in c) for c in _HEX8] + [
        tuple((x + y) / 2 for x, y in zip(_HEX8[a], _HEX8[b]))
        for a, b in edges]
    return [gmsh[i] for i in GMSH_HEX20_TO_SERENDIPITY]


def _opts(formulation: str, geom: str = "linear") -> list[Any]:
    """The flags the bridge emits (its std / linear defaults elided)."""
    out: list[Any] = []
    if formulation != "std":
        out += ["-formulation", formulation]
    if geom != "linear":
        out += ["-geom", geom]
    return out


#: case id -> (class, token, ndm, node count, argv after the node tags,
#: given the carried tag).  One row per formulation (and ``-geom``) the
#: bridge emits for the class with a small-strain ManzariDafalias.
_ELEMENT_CASES: dict[str, tuple[str, str, int, str, Any]] = {}


def _add(cls: str, token: str, ndm: int, coords: str, *,
         formulation: str = "std", geom: str = "linear",
         thick: bool) -> None:
    case = cls if formulation == "std" and geom == "linear" else (
        f"{cls}-{formulation}" + ("" if geom == "linear" else f"-{geom}"))
    tail = _opts(formulation, geom) + (["-thick", 1.0] if thick else [])
    _ELEMENT_CASES[case] = (cls, token, ndm, coords, tail)


for _f in ("std", "bbar", "ssp"):
    _add("LadrunoQuad", "LadrunoQuad", 2, "quad4", formulation=_f, thick=True)
_add("LadrunoCST", "LadrunoCST", 2, "tri3", thick=True)
_add("LadrunoLST", "LadrunoLST", 2, "tri6", thick=True)
for _f in ("std", "bbar", "uri", "ssp", "eas"):
    _add("LadrunoBrick", "LadrunoBrick", 3, "hex8", formulation=_f,
         thick=False)
for _f in ("std", "bbar"):
    _add("LadrunoBrick", "LadrunoBrick", 3, "hex8", formulation=_f,
         geom="corot", thick=False)
for _f in ("std", "uri"):
    _add("LadrunoBrick20", "LadrunoBrick20", 3, "hex20", formulation=_f,
         thick=False)

#: case id -> (host token, ndm, coords, host argv tail, wrapper chain over
#: the MD tag ``t``: a list of ``(token, extra argv)`` applied innermost
#: first, the element carrying the outermost).
_WRAPPER_CASES: dict[str, tuple[str, int, str, list[Any], list[Any]]] = {
    "LogStrain": ("LadrunoBrick", 3, "hex8", ["-geom", "finite"],
                  [("LogStrain", [])]),
    "LogStrain2D": ("LadrunoQuad", 2, "quad4",
                    ["-geom", "finite", "-thick", 1.0],
                    [("LogStrain2D", [])]),
    "InitDefGrad-LogStrain": ("LadrunoBrick", 3, "hex8", ["-geom", "finite"],
                              [("LogStrain", []), ("InitDefGrad", [])]),
    "InitDefGrad-direct": ("LadrunoBrick", 3, "hex8", ["-geom", "finite"],
                           [("InitDefGrad", [])]),
}


def _coords(name: str) -> list[tuple[float, ...]]:
    if name == "hex20":
        return _hex20()
    return {"quad4": _QUAD4, "tri3": _TRI3, "tri6": _TRI6,
            "hex8": _HEX8}[name]  # type: ignore[return-value]


def _run(token: str, ndm: int, coords_name: str, tail: list[Any],
         chain: list[Any], how: str) -> list[float]:
    """Two elements of ``token`` on a homogeneous strain path; their
    stresses at t=2.  ``chain`` wraps each MD tag ``t`` (outer tag
    ``10 * depth + t``); the elements carry the outermost tag."""
    from apeGmsh.opensees.emitter.live import get_ops

    ops = get_ops()
    coords = _coords(coords_name)
    eps = (-1e-2, -1.5e-2, 0.0)[:ndm]
    tags = [1, 2] if how == "reference" else [1, 1]
    ops.wipe()
    try:
        ops.model("basic", "-ndm", ndm, "-ndf", ndm)
        carried: dict[int, int] = {}
        for t in sorted(set(tags)):
            ops.nDMaterial("ManzariDafalias", t, *MD_ARGS)
            inner = t
            for depth, (wtoken, extra) in enumerate(chain, start=1):
                outer = 10 * depth + t
                ops.nDMaterial(wtoken, outer, inner, *extra)
                inner = outer
            carried[t] = inner
        ops.timeSeries("Path", 1, "-time", 0.0, 1.0, 2.0,
                       "-values", 0.0, 1.0, 2.0)
        ops.pattern("Plain", 1, 1)
        for q, t in enumerate(tags):
            nodes = []
            for i, c in enumerate(coords):
                n = 100 * q + i + 1
                xyz = [float(v) for v in c]
                ops.node(n, xyz[0] + 3.0 * q, *xyz[1:])
                nodes.append(n)
                for d in range(ndm):
                    ops.sp(n, d + 1, eps[d] * xyz[d])
            ops.element(token, q + 1, *nodes, carried[t], *tail)
        for t in sorted(set(tags)):
            ops.updateMaterialStage("-material", t, "-stage", 0)
        ops.constraints("Penalty", 1.0e12, 1.0e12)
        ops.numberer("Plain")
        ops.system("FullGeneral")
        ops.test("NormUnbalance", 1e-8, 20, 0)
        ops.algorithm("Linear")
        ops.integrator("LoadControl", 0.05)
        ops.analysis("Static")
        rc = ops.analyze(20)
        if rc != 0:
            raise RuntimeError(f"elastic leg analyze rc={rc}")
        if how == "routed":
            # The two lines stage_emit.MaterialStageCarriers writes for a
            # plastic flip of a material on elements 1 and 2.
            ops.setParameter("-val", 1.0, "-ele", 1, 2, "materialState", "1")
            ops.updateMaterialStage("-material", 1, "-stage", 1)
        else:
            for t in sorted(set(tags)):
                ops.updateMaterialStage("-material", t, "-stage", 1)
        rc = ops.analyze(20)
        if rc != 0:
            raise RuntimeError(f"plastic leg analyze rc={rc}")
        out: list[float] = []
        for e in (1, 2):
            got = ops.eleResponse(e, "stresses") or ops.eleResponse(e, "stress")
            if not got:
                raise RuntimeError(f"element {e}: no stresses response")
            out.extend(float(v) for v in got)
        return out
    finally:
        ops.wipe()


def child_main(kind: str, case: str) -> dict[str, Any]:
    """Run one case's three flips; the child's whole answer."""
    from apeGmsh.opensees.emitter.live import get_backend_info

    if kind == "element":
        _cls, token, ndm, coords, tail = _ELEMENT_CASES[case]
        chain: list[Any] = []
    else:
        token, ndm, coords, tail, chain = _WRAPPER_CASES[case]
    out: dict[str, Any] = {"build": get_backend_info().build}
    for how in ("reference", "routed", "first"):
        try:
            out[how] = _run(token, ndm, coords, tail, chain, how)
        except Exception as exc:  # noqa: BLE001 - recorded, judged by parent
            out[how] = f"{type(exc).__name__}: {exc}"
    return out


def _in_child(kind: str, case: str) -> tuple[int, dict[str, Any] | None, str]:
    """Run ``child_main`` in a fresh interpreter: ``(rc, result, tail)``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_ROOT / "src"), str(_ROOT)]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env.setdefault("LADRUNO_OPENSEES_QUIET", "1")
    code = (
        "import json, sys\n"
        "from tests.opensees.integration import test_md_stage_ladruno_at_pin as m\n"
        f"r = m.child_main({kind!r}, {case!r})\n"
        f"print({_SENTINEL!r} + json.dumps(r), flush=True)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code], cwd=str(_ROOT), env=env,
        capture_output=True, text=True, timeout=300,
    )
    result = None
    for line in done.stdout.splitlines():
        if line.startswith(_SENTINEL):
            result = json.loads(line[len(_SENTINEL):])
    tail = (done.stdout[-1500:] + "\n--- stderr ---\n" + done.stderr[-1500:])
    return done.returncode, result, tail


def _pinned_sha() -> str:
    for line in (_ROOT / "FORK_PIN").read_text(encoding="utf-8").splitlines():
        if line.startswith("sha="):
            return line.split("=", 1)[1].strip()
    raise AssertionError("FORK_PIN has no sha= line")


def _diff(a: list[float], b: list[float]) -> float:
    assert len(a) == len(b), (len(a), len(b))
    return max(abs(x - y) for x, y in zip(a, b))


def _check_build(result: dict[str, Any]) -> None:
    if os.environ.get("APEGMSH_FORK_PIN_ENFORCE") == "1":
        assert result["build"] == _pinned_sha(), (
            f"measured on {result['build']}, not FORK_PIN {_pinned_sha()}")


def _measure(kind: str, case: str) -> dict[str, Any]:
    rc, result, tail = _in_child(kind, case)
    assert result is not None, (
        f"{case}: the child ended without a result (rc={rc}); the fork "
        f"crashed or refused the deck.\n{tail}")
    _check_build(result)
    result["tail"] = tail
    return result


def _verdict(result: dict[str, Any]) -> str:
    """``forwards``, ``drops`` or the child's error text."""
    ref, routed, first = (result[h] for h in ("reference", "routed", "first"))
    for how, got in (("reference", ref), ("routed", routed), ("first", first)):
        if isinstance(got, str):
            return f"{how} failed: {got}"
    if _diff(routed, ref) == 0.0 and _diff(first, ref) > 0.0:
        return "forwards"
    if _diff(routed, first) == 0.0 and _diff(first, ref) > 0.0:
        return "drops"
    return (f"inconclusive: |routed-ref|={_diff(routed, ref):.3e} "
            f"|first-ref|={_diff(first, ref):.3e} "
            f"|routed-first|={_diff(routed, first):.3e}")


@pytest.mark.parametrize("case", sorted(_ELEMENT_CASES))
def test_ladruno_element_forwards_material_state_at_pin(case: str) -> None:
    """Measured at FORK_PIN 100eb9ae (live-fork run 39874892907): every case
    forwards, bit for bit (on 79e062367 every one dropped it)."""
    result = _measure("element", case)
    assert _verdict(result) == "forwards", (
        f"{case} on {result['build']}: {_verdict(result)}\n{result['tail']}")


#: What each wrapper case did at FORK_PIN 100eb9ae (live-fork run
#: 39874892907).  ``crash``: ``LogStrain`` asks its inner for
#: ``getType()``, which ManzariDafalias does not implement, and the
#: NDMaterial base exits the process (rc 255) before any analysis.
#: ``refused``: ``InitDefGrad`` refuses a small-strain inner at
#: construction.  A case that starts to run fails here and asks for a
#: re-measure; the bridge keeps all three off ``MATERIAL_STATE_WRAPPERS``.
_WRAPPER_MEASURED = {
    "LogStrain": "crash",
    "LogStrain2D": "crash",
    "InitDefGrad-LogStrain": "crash",
    "InitDefGrad-direct": "refused",
}
_MD_GETTYPE_EXIT = "ManzariDafalias::getType -- subclass responsibility"


@pytest.mark.parametrize("case", sorted(_WRAPPER_CASES))
def test_ladruno_wrapper_over_md_at_pin(case: str) -> None:
    from apeGmsh.opensees._internal.build.stage_emit import (
        MATERIAL_STATE_WRAPPERS,
    )

    chain = _WRAPPER_CASES[case][4]
    assert not {tok for tok, _extra in chain} & set(MATERIAL_STATE_WRAPPERS)
    rc, result, tail = _in_child("wrapper", case)
    if _WRAPPER_MEASURED[case] == "crash":
        assert result is None and rc != 0, (
            f"{case} no longer ends the process (rc={rc}): re-measure it "
            f"(#2457) and list it on MATERIAL_STATE_WRAPPERS if it forwards."
            f"\n{result}\n{tail}")
        assert _MD_GETTYPE_EXIT in tail, tail
    else:
        assert result is not None, f"{case}: rc={rc}\n{tail}"
        _check_build(result)
        assert all(isinstance(result[h], str) and "OpenSeesError" in result[h]
                   for h in ("reference", "routed", "first")), (
            f"{case} is no longer refused at construction: re-measure it "
            f"(#2457).\n{result}\n{tail}")

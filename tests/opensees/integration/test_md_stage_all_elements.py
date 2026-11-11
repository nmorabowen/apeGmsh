"""#2445 / #2446: the emitted plastic stage flip initialises every element.

``updateMaterialStage`` alone runs ``Elastic2Plastic()`` on the first element
that carries a ``ManzariDafalias`` / ``LadrunoSANISAND`` material
(``MaterialStageParameter::setDomain`` stops there).  The bridge now emits
the flip as ``stage_emit.material_stage_flip_commands``: an element sweep
(``setParameter -val 1 -eleRange ... materialState <tag>``) and then the
``updateMaterialStage`` line.

Each test runs one raw model three ways on the resolved backend, so the
stock lane runs it on stock openseespy and the fork lane on the fork:

* ``emitted``: one material on every element, flipped by the lines the
  ``PyEmitter`` renders for ``update_material_stage`` (executed as written);
* ``reference``: each element its own material tag and its own
  ``updateMaterialStage`` (each runs its own ``Elastic2Plastic()``);
* ``first``: one material, the bare ``updateMaterialStage`` line the bridge
  emitted before #2446.

``emitted`` must equal ``reference`` to 1e-10 (measured bit for bit), and
``first`` must differ from it, so the model can tell the two flips apart:
revert the emit and the first assertion fails.

The tag-blind neighbour test runs each leg in a child interpreter (#2881):
the plastic leg of stock ``StressDensityModel`` can end in a Fortran
``STOP``, which is ``exit(0)``, and in the pytest process that ended the
``live-stock`` job with status 0 and no report (#2543, as #2542 found for
the sibling ``test_tag_blind_stage_parameter.py``).
"""
from __future__ import annotations

import io
import sys
from typing import Any, Callable, NamedTuple

import pytest

from apeGmsh.opensees.emitter.py import PyEmitter

#: Dafalias & Manzari (2004) Toyoura sand, the OpenSees example set.
MD_ARGS = [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0, 0.01,
           7.05, 0.968, 1.1, 0.704, 3.5, 4.0, 600.0, 1.42]


_FLIP_PREFIXES = ("ops.setParameter(", "ops.updateMaterialStage(")


def _flip_lines_of(em: PyEmitter) -> list[str]:
    buf = io.StringIO()
    em.write_to(buf)
    return [ln for ln in buf.getvalue().splitlines()
            if ln.startswith(_FLIP_PREFIXES)]


def emitted_flip_lines(
    mat_tag: int, stage: int = 1, *, emitter: "PyEmitter | None" = None,
) -> list[str]:
    """The openseespy lines ``emitter`` writes for one flip, after the
    model it has seen so far (a fresh emitter has seen no element)."""
    em = PyEmitter() if emitter is None else emitter
    before = len(_flip_lines_of(em))
    em.update_material_stage(mat_tag, stage)
    lines = _flip_lines_of(em)[before:]
    assert lines, "the PyEmitter wrote no flip line"
    return lines


class Deck:
    """The resolved backend, with every ``nDMaterial`` and ``element``
    command also handed to a :class:`PyEmitter`, so an emitted flip comes
    from the emitter's own record of which elements carry the material."""

    def __init__(self, ops: Any) -> None:
        self.ops = ops
        self.emitter = PyEmitter()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.ops, name)

    def wipe(self) -> None:
        self.ops.wipe()
        self.emitter = PyEmitter()

    def nDMaterial(self, cls: str, tag: int, *args: Any) -> None:  # noqa: N802
        self.ops.nDMaterial(cls, tag, *args)
        self.emitter.nDMaterial(cls, tag, *args)

    def element(self, token: str, tag: int, *args: Any) -> None:
        self.ops.element(token, tag, *args)
        self.emitter.element(token, tag, *args)

    def emitted_flip(self, mat_tag: int, stage: int = 1) -> list[str]:
        lines = emitted_flip_lines(mat_tag, stage, emitter=self.emitter)
        for line in lines:
            exec(line, {"ops": self.ops})  # noqa: S102 - the emitted line
        return lines


def _ops() -> Deck:
    """The resolved backend: the fork where it imports (the live-fork lane
    binds it as ``opensees``, with no openseespy), else stock openseespy
    (#2503). Skips only when no backend imports at all."""
    from apeGmsh.opensees.emitter.live import get_ops

    try:
        return Deck(get_ops())
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


def flip(ops: Deck, how: str, tags: "list[int]") -> None:
    """Flip to stage 1: ``how`` is emitted / reference / first."""
    if how == "emitted":
        (tag,) = set(tags)
        ops.emitted_flip(tag)
    elif how == "reference":
        for tag in sorted(set(tags)):
            ops.updateMaterialStage("-material", tag, "-stage", 1)
    elif how == "first":
        (tag,) = set(tags)
        ops.updateMaterialStage("-material", tag, "-stage", 1)
    else:  # pragma: no cover
        raise ValueError(how)


def _chain(ops: Any, tol: float, dlam: float) -> None:
    ops.constraints("Transformation")
    ops.numberer("RCM")
    ops.system("UmfPack")
    ops.test("NormUnbalance", tol, 200, 0)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", dlam)
    ops.analysis("Static")


def _state(ops: Any, eles: "list[int]") -> list[float]:
    out: list[float] = []
    for n in sorted(ops.getNodeTags()):
        out += list(ops.nodeDisp(n))
    for e in eles:
        out += list(ops.eleResponse(e, "stress"))
    return out


def two_quads(ops: Any, how: str, *, mat: str, element: str,
              tol: float = 1e-8, n_plastic: int = 20,
              extra: float = 20.0) -> "tuple[int, list[float]]":
    """Two disconnected unit quads, plane strain: elastic hydrostatic
    100 kPa, flip, then ``2 * extra`` kPa more on sigma_y."""
    ops.wipe()
    up = element == "LadrunoUP"
    ndf = 3 if up else 2
    ops.model("basic", "-ndm", 2, "-ndf", ndf)
    p = [1] if up else []
    eles = [1, 2]
    tags = [1, 2] if how == "reference" else [1, 1]
    for t in sorted(set(tags)):
        ops.nDMaterial(mat, t, *MD_ARGS)
    for q, (e, t) in enumerate(zip(eles, tags)):
        b, x0 = 10 * q, 3.0 * q
        for i, (dx, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
            ops.node(b + i + 1, x0 + dx, float(y))
        ops.fix(b + 1, 1, 1, *p)
        ops.fix(b + 2, 0, 1, *p)
        ops.fix(b + 4, 1, 0, *p)
        if up:
            ops.fix(b + 3, 0, 0, 1)
        conn = (b + 1, b + 2, b + 3, b + 4)
        if element == "SSPquad":
            ops.element("SSPquad", e, *conn, t, "PlaneStrain", 1.0)
        elif element == "quad":
            ops.element("quad", e, *conn, 1.0, "PlaneStrain", t)
        else:
            ops.element("LadrunoUP", e, *conn, t, "-Kf", 2.2e6, "-poro",
                        0.4, "-rhoF", 1.0, "-perm", 1e-4, 1e-4)
    for t in sorted(set(tags)):
        ops.updateMaterialStage("-material", t, "-stage", 0)
    z = [0.0] if up else []
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for q in range(2):
        b = 10 * q
        ops.load(b + 3, -50.0, -50.0, *z)
        ops.load(b + 4, 0.0, -50.0, *z)
        ops.load(b + 2, -50.0, 0.0, *z)
    _chain(ops, tol, 0.05)
    code = int(ops.analyze(20))
    ops.loadConst("-time", 0.0)
    flip(ops, how, tags)
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for q in range(2):
        for n in (3, 4):
            ops.load(10 * q + n, 0.0, -extra, *z)
    ops.integrator("LoadControl", 1.0 / n_plastic)
    code = code or int(ops.analyze(n_plastic))
    state = _state(ops, eles)
    ops.wipe()
    return code, state


def brick_column(ops: Any, how: str, *, mat: str = "ManzariDafalias",
                 ) -> "tuple[int, list[float]]":
    """Four SSPbricks stacked, laterally fixed (K0): self-weight plus a
    100 kPa surcharge, flip, then 40 kPa more on top.  Each brick sits at
    its own depth, so each initialises from its own stress."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    eles = [1, 2, 3, 4]
    tags = eles if how == "reference" else [1, 1, 1, 1]
    for t in sorted(set(tags)):
        ops.nDMaterial(mat, t, *MD_ARGS)
    for k in range(5):
        for j, (x, y) in enumerate(corners):
            n = 100 * k + j + 1
            ops.node(n, float(x), float(y), float(k))
            ops.fix(n, 1, 1, 1 if k == 0 else 0)
    for k, (e, t) in enumerate(zip(eles, tags)):
        conn = [100 * k + j + 1 for j in range(4)]
        conn += [100 * (k + 1) + j + 1 for j in range(4)]
        ops.element("SSPbrick", e, *conn, t)
    for t in sorted(set(tags)):
        ops.updateMaterialStage("-material", t, "-stage", 0)
    weight = 9.81 * MD_ARGS[-1]
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for k in range(1, 5):
        w = weight * (0.5 if k == 4 else 1.0) / 4.0
        for j in range(4):
            ops.load(100 * k + j + 1, 0.0, 0.0, -w - (25.0 if k == 4 else 0.0))
    _chain(ops, 1e-8, 0.05)
    code = int(ops.analyze(20))
    ops.loadConst("-time", 0.0)
    flip(ops, how, tags)
    ops.timeSeries("Linear", 2)
    ops.pattern("Plain", 2, 2)
    for j in range(4):
        ops.load(400 + j + 1, 0.0, 0.0, -10.0)
    code = code or int(ops.analyze(20))
    state = _state(ops, eles)
    ops.wipe()
    return code, state


def _max_diff(a: "list[float]", b: "list[float]") -> float:
    assert len(a) == len(b)
    return max(abs(x - y) for x, y in zip(a, b))


def assert_route_matches_reference(
    run: "Callable[[str], tuple[int, list[float]]]", *, first_fails: bool = False,
) -> None:
    code_r, ref = run("reference")
    code_e, emitted = run("emitted")
    code_f, first = run("first")
    assert code_r == 0 and code_e == 0
    scale = max(1.0, max(abs(v) for v in ref))
    assert _max_diff(emitted, ref) <= 1e-10 * scale
    # The model separates the two flips: the old one is wrong here, or
    # (LadrunoSANISAND) cannot carry the plastic leg at all.
    if first_fails:
        assert code_f != 0 or _max_diff(first, ref) > 1e-8 * scale
    else:
        assert code_f == 0
        assert _max_diff(first, ref) > 1e-8 * scale


@pytest.mark.live
@pytest.mark.parametrize("element", ["SSPquad"])
def test_two_quads_manzari(element: str) -> None:
    ops = _ops()
    assert_route_matches_reference(
        lambda how: two_quads(ops, how, mat="ManzariDafalias",
                              element=element))


@pytest.mark.live
def test_brick_column_manzari() -> None:
    ops = _ops()
    assert_route_matches_reference(lambda how: brick_column(ops, how))


@pytest.mark.live
@pytest.mark.ladruno_fork
def test_two_quads_ladruno_sanisand() -> None:
    ops = _ops()
    assert_route_matches_reference(
        lambda how: two_quads(ops, how, mat="LadrunoSANISAND",
                              element="SSPquad", tol=1e-4, n_plastic=25,
                              extra=5.0),
        first_fails=True)


@pytest.mark.live
@pytest.mark.ladruno_fork
@pytest.mark.parametrize("mat", ["ManzariDafalias", "LadrunoSANISAND"])
def test_two_ladruno_up_elements(mat: str) -> None:
    """LadrunoUP forwards ``materialState``; the ``updateMaterialStage``
    line that follows the sweep still refreshes its stage caches."""
    ops = _ops()
    sanisand = mat == "LadrunoSANISAND"
    assert_route_matches_reference(
        lambda how: two_quads(ops, how, mat=mat, element="LadrunoUP",
                              tol=1e-4 if sanisand else 1e-8,
                              n_plastic=25 if sanisand else 20,
                              extra=5.0 if sanisand else 20.0),
        first_fails=sanisand)


@pytest.mark.live
def test_four_node_quad_drops_the_sweep() -> None:
    """FourNodeQuad reads ``materialState`` as a per-point address, so the
    sweep misses it and the emitted flip is the first-element-only one:
    why ``stage/_builder.py`` refuses a multi-element plastic flip on it."""
    ops = _ops()
    _code_r, ref = two_quads(ops, "reference", mat="ManzariDafalias",
                             element="quad")
    _code_e, emitted = two_quads(ops, "emitted", mat="ManzariDafalias",
                                 element="quad")
    _code_f, first = two_quads(ops, "first", mat="ManzariDafalias",
                               element="quad")
    assert _max_diff(emitted, first) == 0.0
    assert _max_diff(emitted, ref) > 1e-6


#: Plastic neighbours whose ``setParameter`` accepts ``materialState``
#: WITHOUT checking the material tag (``DruckerPrager.cpp`` at the
#: ``"materialState"`` branch; ``StressDensityModel.cpp`` likewise), as
#: ``(class, args, strain amplitude)``.  A flip that offered
#: ``materialState`` to their elements would switch them too.
#: ``BoundingCamClay`` is tag-blind as well, but it cannot run from a
#: zero stress on openseespy (NaN on the fork, an access violation on
#: stock 3.8.0), and ``stressDensity`` is registered for Tcl only; the
#: deck tests of ``test_md_stage_route_elements.py`` pin that the flip
#: names the carriers and nothing else.
TAG_BLIND_NEIGHBOURS = [
    ("DruckerPrager",
     [1e5, 5e4, 10.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 1000.0, 1.0], 1.0),
    ("StressDensityModel",
     [2.0, 0.65, 600.0, 0.6, 0.3, 0.58, 0.023, 230.0, 65.0, 79.0, 16.0,
      4.0, 0.22, 0.0, 0.0055, 0.607, 98.1], 0.05),
]

_TRI6 = [(0, 0), (1, 0), (0, 1), (0.5, 0), (0.5, 0.5), (0, 0.5)]
_QUAD4 = [(0, 0), (1, 0), (1, 1), (0, 1)]


def _neighbour_model(
    ops: Deck, how: str,
    neighbours: "list[tuple[str, list[float], float]]" = TAG_BLIND_NEIGHBOURS,
    *, on_flip: "Callable[[], None] | None" = None,
) -> "tuple[int, list[list[float]]]":
    """Every DOF on a prescribed strain path to t=1, the flip, then on to
    t=2.  Elements 1 and 2: SSPquads carrying ManzariDafalias (tags 1 and
    2 for the reference, tag 1 otherwise).  Then one SixNodeTri per
    ``neighbours`` class in simple shear, an elastic quad and a truss.
    ``on_flip`` runs right after the flip.  Returns each element's
    response."""
    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    tags = [1, 2] if how == "reference" else [1, 1]
    for t in sorted(set(tags)):
        ops.nDMaterial("ManzariDafalias", t, *MD_ARGS)
    ops.timeSeries("Path", 1, "-time", 0.0, 1.0, 2.0,
                   "-values", 0.0, 1.0, 2.0)
    ops.pattern("Plain", 1, 1)
    origin = [0.0]

    def place(coords: "list[tuple[int, int]]",
              path: "Callable[[float, float], tuple[float, float]]",
              ) -> list[int]:
        x0 = origin[0]
        origin[0] += 3.0
        first = len(ops.getNodeTags()) + 1
        out = []
        for i, (x, y) in enumerate(coords):
            n = first + i
            ops.node(n, x0 + x, float(y))
            ux, uy = path(float(x), float(y))
            ops.sp(n, 1, ux)
            ops.sp(n, 2, uy)
            out.append(n)
        return out

    def squeeze(x: float, y: float) -> tuple[float, float]:
        return -1e-2 * x, -1.5e-2 * y

    for e, t in zip((1, 2), tags):
        ops.element("SSPquad", e, *place(_QUAD4, squeeze), t,
                    "PlaneStrain", 1.0)
    for k, (cls, args, amp) in enumerate(neighbours):
        ops.nDMaterial(cls, 10 + k, *args)

        def shear(x: float, y: float, amp: float = amp) -> tuple[float, float]:
            return amp * (1e-2 * y - 2e-3 * x), amp * (-2e-3 * y)

        ops.element("tri6n", 3 + k, *place(_TRI6, shear), 1.0,
                    "PlaneStrain", 10 + k)
    n = len(neighbours)
    ops.nDMaterial("ElasticIsotropic", 50, 1.0e5, 0.3)
    ops.element("quad", 3 + n, *place(_QUAD4, squeeze), 1.0, "PlaneStrain",
                50)
    ops.uniaxialMaterial("Elastic", 60, 1.0e5)
    ops.element("Truss", 4 + n, *place([(0, 0), (1, 0)], squeeze), 1.0, 60)
    for t in sorted(set(tags)):
        ops.updateMaterialStage("-material", t, "-stage", 0)
    ops.constraints("Penalty", 1.0e12, 1.0e12)
    ops.numberer("Plain")
    ops.system("FullGeneral")
    ops.test("NormUnbalance", 1e-8, 20, 0)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 0.05)
    ops.analysis("Static")
    code = int(ops.analyze(20))
    if how == "every_element":
        # The sweep fd836151d emitted, kept here as the counter-example.
        ops.setParameter("-val", 1.0, "-eleRange", 1, 2147483647,
                         "materialState", "1")
        ops.updateMaterialStage("-material", 1, "-stage", 1)
    else:
        flip(ops, how, tags)
    if on_flip is not None:
        on_flip()
    code = code or int(ops.analyze(20))
    out = [
        list(ops.eleResponse(e, "stresses") or ops.eleResponse(e, "stress")
             or ops.eleResponse(e, "axialForce"))
        for e in range(1, 5 + n)
    ]
    ops.wipe()
    return code, out


#: The child that runs one leg of :func:`_neighbour_model` with one
#: neighbour class.  It prints ``BACKEND`` first, ``FLIPPED`` right after
#: the flip and ``RESULT`` last, so a child that died in between is
#: visible as such, and where.
_SWEEP_CHILD = r"""
import json, sys
from apeGmsh.opensees.emitter.live import get_backend_name, get_ops
from tests.opensees.integration import test_md_stage_all_elements as m

spec = json.loads(sys.argv[1])
(neighbour,) = [nb for nb in m.TAG_BLIND_NEIGHBOURS if nb[0] == spec["cls"]]
ops = get_ops()
print("BACKEND", get_backend_name(), flush=True)
code, out = m._neighbour_model(
    m.Deck(ops), spec["how"], [neighbour],
    on_flip=lambda: print("FLIPPED", flush=True))
print("RESULT", json.dumps({"code": code, "out": out}), flush=True)
"""

#: What the stress-density Fortran (``SDM-UC.f``) writes before each of its
#: ``STOP`` statements.  Only the plastic leg (stage 1) calls that Fortran.
_SDM_STOP = "Stop in S-D model"

_LEGS = ("reference", "emitted", "every_element")


class _Leg(NamedTuple):
    backend: str
    returncode: int
    stdout: str
    log: str  # return code, stdout and stderr, for assertion messages
    code: "int | None"  # the analyze code; None when RESULT never printed
    out: "list[list[float]] | None"  # each element's response


def _run_leg(cls: str, how: str) -> _Leg:
    """``_neighbour_model(how)`` with the one neighbour ``cls``, in a child
    interpreter on the resolved backend."""
    import json
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    env = {**os.environ, "LADRUNO_OPENSEES_QUIET": "1",
           "PYTHONPATH": os.pathsep.join(
               [str(root / "src"), str(root),
                os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    proc = subprocess.run(
        [sys.executable, "-c", _SWEEP_CHILD,
         json.dumps({"cls": cls, "how": how})],
        cwd=str(root), capture_output=True, text=True, timeout=300, env=env,
    )
    log = f"{cls} {how}: rc={proc.returncode}\n{proc.stdout}\n{proc.stderr}"
    backend = [ln.split()[1] for ln in proc.stdout.splitlines()
               if ln.startswith("BACKEND ")]
    assert backend, f"the child never resolved a backend:\n{log}"
    result = [json.loads(ln[len("RESULT "):]) for ln in proc.stdout.splitlines()
              if ln.startswith("RESULT ")]
    code, out = ((int(result[0]["code"]), result[0]["out"]) if result
                 else (None, None))
    return _Leg(backend[0], proc.returncode, proc.stdout, log, code, out)


def _backend() -> str:
    """The backend this process resolves; the children must resolve the
    same one.  Skips only when no backend imports at all."""
    from apeGmsh.opensees.emitter.live import get_backend_name

    try:
        return get_backend_name()
    except ImportError as e:
        pytest.skip(f"no OpenSees backend: {e}")


@pytest.mark.live
@pytest.mark.parametrize("cls", [c for c, _a, _amp in TAG_BLIND_NEIGHBOURS])
def test_the_sweep_leaves_other_elements_alone(
    cls: str, record_property: "Callable[[str, object], None]",
) -> None:
    """The emitted flip names the carriers only.  A DruckerPrager or a
    StressDensityModel element in plastic shear beside the sand keeps the
    per-element reference's answer, which an every-element sweep breaks
    (it switches them: measured on stock 3.8.0 and the fork).

    Each leg runs in a child (#2881).  On the stock Linux wheel the swept
    ``StressDensityModel`` may instead reach the model's Fortran ``STOP``
    (#2543): the switched element enters the plastic leg, whose ``strcon``
    LAPACK answers.  That STOP after the flip is the switch, and it is the
    one early death accepted; the test records which outcome it took."""
    backend = _backend()
    legs = {how: _run_leg(cls, how) for how in _LEGS}
    for leg in legs.values():
        assert leg.backend == backend, leg.log
    ref, emitted, swept = (legs[how] for how in _LEGS)
    for leg in (ref, emitted):
        assert leg.returncode == 0 and leg.code == 0, leg.log
        assert leg.out is not None and _SDM_STOP not in leg.log, leg.log
    assert ref.out is not None and emitted.out is not None
    flat_ref = [v for resp in ref.out for v in resp]
    scale = max(1.0, max(abs(v) for v in flat_ref))
    assert _max_diff([v for r in emitted.out for v in r],
                     flat_ref) <= 1e-10 * scale, (emitted.log, ref.log)
    if swept.out is None:
        # Only the plastic leg calls the stress-density Fortran, so its STOP
        # after the flip is the switch; nothing else may end the child early.
        # A Fortran STOP exits 0: a signal or a nonzero code is a crash.
        assert cls == "StressDensityModel" and backend == "stock-openseespy", \
            swept.log
        assert swept.returncode == 0, swept.log
        assert "FLIPPED" in swept.stdout and _SDM_STOP in swept.stdout, \
            swept.log
        assert swept.stdout.index("FLIPPED") < swept.stdout.index(_SDM_STOP), \
            swept.log
        outcome = "stop-after-flip"
    else:
        assert swept.returncode == 0 and swept.code == 0, swept.log
        hit = _max_diff(swept.out[2], ref.out[2])
        assert hit > 1e-3 * max(abs(v) for v in ref.out[2]), (cls, swept.log)
        outcome = "switched"
    record_property("outcome", outcome)
    print(f"{cls} on {backend}: the swept leg {outcome}")


# ---------------------------------------------------------------------------
# The neighbour test against stubbed children (#2881): what it accepts
# ---------------------------------------------------------------------------

_STOP_LINE = " ### / Stop in S-D model / ### cnjpnt: no conjugate point(1)\n"
_STUB_REF = [[1.0, 2.0, 3.0]] * 5
_STUB_SWEPT = [[1.0, 2.0, 3.0]] * 2 + [[10.0, 20.0, 30.0]] + [[1.0, 2.0, 3.0]] * 2


def _stub_ok(out: "list[list[float]]") -> "tuple[int, str, str]":
    import json

    return (0, "FLIPPED\nRESULT " + json.dumps({"code": 0, "out": out}) + "\n",
            "")


_STOP_DEATH = (0, "FLIPPED\n" + _STOP_LINE, "")
_GOOD = {"reference": _stub_ok(_STUB_REF), "emitted": _stub_ok(_STUB_REF),
         "every_element": _stub_ok(_STUB_SWEPT)}


def _with(**legs: "tuple[int, str, str]") -> "dict[str, tuple[int, str, str]]":
    return {**_GOOD, **legs}


@pytest.mark.parametrize(("cls", "backend", "children", "outcome"), [
    ("DruckerPrager", "stock-openseespy", _GOOD, "switched"),
    ("StressDensityModel", "ladruno-fork", _GOOD, "switched"),
    ("StressDensityModel", "stock-openseespy",
     _with(every_element=_STOP_DEATH), "stop-after-flip"),
    # A wrong value: the emitted flip off the reference, or a sweep that
    # did not switch the neighbour.
    ("DruckerPrager", "stock-openseespy",
     _with(emitted=_stub_ok([[1.0, 2.0, 3.0 + 1e-6]] + _STUB_REF[1:])), None),
    ("StressDensityModel", "ladruno-fork",
     _with(every_element=_stub_ok(_STUB_REF)), None),
    # A nonzero analyze code.
    ("DruckerPrager", "stock-openseespy",
     _with(every_element=(0, "FLIPPED\nRESULT "
                          '{"code": -3, "out": [[1.0]]}\n', "")), None),
    # The STOP accepted only for StressDensityModel, only on stock.
    ("DruckerPrager", "stock-openseespy",
     _with(every_element=_STOP_DEATH), None),
    ("StressDensityModel", "ladruno-fork",
     _with(every_element=_STOP_DEATH), None),
    # ... only on the swept leg.
    ("StressDensityModel", "stock-openseespy",
     _with(emitted=_STOP_DEATH), None),
    ("StressDensityModel", "stock-openseespy",
     _with(reference=_STOP_DEATH), None),
    # ... only as a clean exit, after the flip, on stdout.
    ("StressDensityModel", "stock-openseespy",
     _with(every_element=(-11,) + _STOP_DEATH[1:]), None),
    ("StressDensityModel", "stock-openseespy",
     _with(every_element=(1,) + _STOP_DEATH[1:]), None),
    ("StressDensityModel", "stock-openseespy",
     _with(every_element=(0, _STOP_LINE + "FLIPPED\n", "")), None),
    ("StressDensityModel", "stock-openseespy",
     _with(every_element=(0, "FLIPPED\n", _STOP_LINE)), None),
    ("StressDensityModel", "stock-openseespy",
     _with(every_element=(0, "FLIPPED\n", "")), None),
], ids=["dp-switched", "sdm-fork-switched", "sdm-stock-stop-after-flip",
        "emitted-off-reference", "sweep-did-not-switch", "nonzero-code",
        "dp-stop", "fork-stop", "stop-in-emitted", "stop-in-reference",
        "signal", "nonzero-exit", "stop-before-flip", "stop-on-stderr",
        "silent-death"])
def test_the_neighbour_test_accepts_only_its_two_outcomes(
    monkeypatch: pytest.MonkeyPatch, cls: str, backend: str,
    children: "dict[str, tuple[int, str, str]]", outcome: "str | None",
) -> None:
    import json
    import subprocess

    def fake_run(argv: "list[str]",
                 **_kw: Any) -> "subprocess.CompletedProcess[str]":
        spec = json.loads(argv[-1])
        assert spec["cls"] == cls
        rc, out, err = children[spec["how"]]
        return subprocess.CompletedProcess(
            argv, rc, f"BACKEND {backend}\n{out}", err)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys.modules[__name__], "_backend", lambda: backend)
    recorded: "dict[str, object]" = {}
    run = test_the_sweep_leaves_other_elements_alone
    if outcome is None:
        with pytest.raises(AssertionError):
            run(cls, recorded.__setitem__)
    else:
        run(cls, recorded.__setitem__)
        assert recorded == {"outcome": outcome}


def test_a_child_on_another_backend_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Children that would pass (``dp-switched``), but on stock while this
    process resolved the fork."""
    import json
    import subprocess

    def fake_run(argv: "list[str]",
                 **_kw: Any) -> "subprocess.CompletedProcess[str]":
        rc, out, err = _GOOD[json.loads(argv[-1])["how"]]
        return subprocess.CompletedProcess(
            argv, rc, "BACKEND stock-openseespy\n" + out, err)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys.modules[__name__], "_backend",
                        lambda: "ladruno-fork")
    with pytest.raises(AssertionError):
        test_the_sweep_leaves_other_elements_alone(
            "DruckerPrager", lambda *_a: None)

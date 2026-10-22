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
"""
from __future__ import annotations

import io
from typing import Any, Callable

import pytest

from apeGmsh.opensees.emitter.py import PyEmitter

pytestmark = pytest.mark.live

#: Dafalias & Manzari (2004) Toyoura sand, the OpenSees example set.
MD_ARGS = [125.0, 0.05, 0.8, 1.25, 0.712, 0.019, 0.934, 0.7, 100.0, 0.01,
           7.05, 0.968, 1.1, 0.704, 3.5, 4.0, 600.0, 1.42]


def _ops() -> Any:
    pytest.importorskip("openseespy.opensees")
    from apeGmsh.opensees.emitter.live import get_ops

    return get_ops()


def emitted_flip_lines(mat_tag: int, stage: int = 1) -> list[str]:
    """The openseespy lines the deck emitter writes for one flip."""
    em = PyEmitter()
    em.update_material_stage(mat_tag, stage)
    buf = io.StringIO()
    em.write_to(buf)
    lines = [
        ln for ln in buf.getvalue().splitlines()
        if ln.startswith(("ops.setParameter(", "ops.updateMaterialStage("))
    ]
    assert lines, "the PyEmitter wrote no flip line"
    return lines


def flip(ops: Any, how: str, tags: "list[int]") -> None:
    """Flip to stage 1: ``how`` is emitted / reference / first."""
    if how == "emitted":
        (tag,) = set(tags)
        for line in emitted_flip_lines(tag):
            exec(line, {"ops": ops})  # noqa: S102 - the emitted deck line
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


@pytest.mark.parametrize("element", ["SSPquad"])
def test_two_quads_manzari(element: str) -> None:
    ops = _ops()
    assert_route_matches_reference(
        lambda how: two_quads(ops, how, mat="ManzariDafalias",
                              element=element))


def test_brick_column_manzari() -> None:
    ops = _ops()
    assert_route_matches_reference(lambda how: brick_column(ops, how))


@pytest.mark.ladruno_fork
def test_two_quads_ladruno_sanisand() -> None:
    ops = _ops()
    assert_route_matches_reference(
        lambda how: two_quads(ops, how, mat="LadrunoSANISAND",
                              element="SSPquad", tol=1e-4, n_plastic=25,
                              extra=5.0),
        first_fails=True)


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


def test_the_sweep_leaves_other_elements_alone() -> None:
    """The sweep offers ``materialState`` to every element in the domain.
    A truss, a zeroLength and an elastic quad beside the sand decline it,
    and the answer is still the per-element reference."""
    ops = _ops()

    def run(how: str) -> "tuple[int, list[float]]":
        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 2)
        tags = [1, 2] if how == "reference" else [1, 1]
        for t in sorted(set(tags)):
            ops.nDMaterial("ManzariDafalias", t, *MD_ARGS)
        ops.nDMaterial("ElasticIsotropic", 50, 1.0e5, 0.3)
        ops.uniaxialMaterial("Elastic", 60, 1.0e5)
        for q, t in enumerate(tags):
            b, x0 = 10 * q, 3.0 * q
            for i, (dx, y) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                ops.node(b + i + 1, x0 + dx, float(y))
            ops.fix(b + 1, 1, 1)
            ops.fix(b + 2, 0, 1)
            ops.fix(b + 4, 1, 0)
            ops.element("SSPquad", q + 1, b + 1, b + 2, b + 3, b + 4, t,
                        "PlaneStrain", 1.0)
        for i, (x, y) in enumerate(((7, 0), (8, 0), (8, 1), (7, 1))):
            ops.node(71 + i, float(x), float(y))
        ops.fix(71, 1, 1)
        ops.fix(72, 0, 1)
        ops.fix(74, 1, 0)
        ops.element("quad", 7, 71, 72, 73, 74, 1.0, "PlaneStrain", 50)
        ops.element("Truss", 8, 72, 73, 1.0, 60)
        ops.node(75, 8.0, 1.0)
        ops.fix(75, 1, 1)
        ops.element("zeroLength", 9, 73, 75, "-mat", 60, 60, "-dir", 1, 2)
        for t in sorted(set(tags)):
            ops.updateMaterialStage("-material", t, "-stage", 0)
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        for b in (0, 10):
            ops.load(b + 3, -50.0, -50.0)
            ops.load(b + 4, 0.0, -50.0)
            ops.load(b + 2, -50.0, 0.0)
        ops.load(73, 0.0, -10.0)
        _chain(ops, 1e-8, 0.05)
        code = int(ops.analyze(20))
        ops.loadConst("-time", 0.0)
        flip(ops, how, tags)
        ops.timeSeries("Linear", 2)
        ops.pattern("Plain", 2, 2)
        for b in (0, 10):
            for n in (3, 4):
                ops.load(b + n, 0.0, -20.0)
        code = code or int(ops.analyze(20))
        state = _state(ops, [1, 2, 7])
        ops.wipe()
        return code, state

    assert_route_matches_reference(run)

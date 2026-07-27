"""Live runs of the newly-added ``system`` solvers.

Gated by the ``live`` marker, and skipped whole when no OpenSees backend
resolves at all. Each test drives a solver in its appropriate regime and
asserts a clean solve:

* ``Diagonal`` — explicit transient with lumped mass (its only valid
  regime; it cannot solve a coupled stiffness matrix).
* ``SProfileSPD`` / ``SparseSYM`` — implicit static cantilever; the tip
  displacement matches the ``BandGeneral`` reference.
* ``Pardiso`` — same cantilever, but skipped unless the bound build
  actually registers it (Ladruno fork + Intel MKL, fork ADR-75 P1).

The parallel-family solvers (``MPIDiagonal`` / ``ParallelProfileSPD``)
are not exercised here — they need an ``OpenSeesMP`` build.
"""
from __future__ import annotations

from typing import cast

import pytest

from apeGmsh.opensees import apeSees
from apeGmsh.opensees.emitter.live import LiveOpsEmitter, get_backend_name

from tests.opensees.fixtures.fem_stub import (
    make_two_node_beam,
)

try:
    # Resolve the backend the tests actually run on — the same one
    # LiveOpsEmitter binds (APEGMSH_OPENSEES_BIN → fork ``opensees`` →
    # stock ``openseespy``). Probing ``openseespy`` directly would read a
    # different build than the one under test.
    get_backend_name()
except ImportError as _e:  # no OpenSees backend on this box at all
    pytest.skip(
        f"no OpenSees backend resolved: {_e}", allow_module_level=True,
    )


def _cantilever(ops: apeSees) -> None:
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()


@pytest.mark.parametrize("system_name", ["SProfileSPD", "SparseSYM"])
@pytest.mark.live
def test_implicit_solver_static_cantilever(system_name: str) -> None:
    expected = 1000.0 * 1.0**3 / (3.0 * 200e9 * 1e-4)

    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    getattr(ops.system, system_name)()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    assert emitter.ops.nodeDisp(2, 1) == pytest.approx(expected, rel=1e-3)


def _build_has_pardiso() -> bool:
    """True iff the bound backend registers ``system Pardiso``.

    Only a Ladruno fork build configured with Intel MKL does (fork
    ADR-75 P1 / PR #623); every other build rejects the token with
    "unknown system type", which the Python interpreter raises as
    ``OpenSeesError`` (verified against the fork build: an unknown name
    raises, ``Pardiso`` returns cleanly).

    Probes the module :class:`LiveOpsEmitter` actually binds — a box with
    both the fork and stock ``openseespy`` installed would otherwise skip
    on stock's answer while the test ran on the fork.
    """
    ops = LiveOpsEmitter(wipe=True).ops
    ops.model("basic", "-ndm", 2, "-ndf", 2)
    try:
        ops.system("Pardiso")
    except Exception:
        return False
    finally:
        ops.wipe()
    return True


#: All three ``-matrixType`` modes are valid on this cantilever — an elastic
#: beam has a symmetric positive-definite tangent, so half-storage reads a
#: matrix that really is symmetric (fork ADR-75 P1d).
@pytest.mark.parametrize(
    ("matrix_type", "krylov", "stats"),
    [
        ("unsymmetric", None, False),
        ("symmetric", None, False),
        ("spd", None, False),
        ("symmetric", None, True),
        # -krylov rides mtype 11 and 2, never -matrixType 2 (fork ADR-75
        # P1e); the primitive refuses that pairing, so only these two.
        ("unsymmetric", 6, False),
        ("spd", 6, False),
    ],
)
@pytest.mark.live
def test_pardiso_static_cantilever(
    matrix_type: str, krylov: "int | None", stats: bool,
) -> None:
    """``system Pardiso`` solves the same cantilever as the reference.

    PARDISO is a drop-in for ``UmfPack``, so the tip displacement must
    match the analytic value in every ``matrix_type`` — only the storage
    and the factorization change.
    """
    if not _build_has_pardiso():
        pytest.skip(
            "this openseespy build has no 'system Pardiso' — needs the "
            "Ladruno fork built with Intel MKL (fork ADR-75 P1)"
        )

    expected = 1000.0 * 1.0**3 / (3.0 * 200e9 * 1e-4)

    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.RCM()
    ops.system.Pardiso(matrix_type=matrix_type, krylov=krylov, stats=stats)
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    ops.integrator.LoadControl(dlam=1.0)
    ops.analysis.Static()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=1) == 0
    assert emitter.ops.nodeDisp(2, 1) == pytest.approx(expected, rel=1e-3)


@pytest.mark.live
def test_pardiso_matrix_types_agree_with_umfpack() -> None:
    """Every PARDISO mode reproduces UmfPack to full double precision.

    This is the claim that makes half-storage usable at all: the fork
    measured rel err 0.0 across modes, and a mismatch here would mean
    apeGmsh mis-emitted ``-matrixType`` (e.g. as a string, which the fork
    degrades to unsymmetric with only a warning).
    """
    if not _build_has_pardiso():
        pytest.skip("this openseespy build has no 'system Pardiso'")

    def _tip(make_system: "object") -> float:
        ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
        ops.model(ndm=3, ndf=6)
        transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
        ops.element.elasticBeamColumn(
            pg="Cols", transf=transf,
            A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
        )
        ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
        ts = ops.timeSeries.Linear()
        with ops.pattern.Plain(series=ts) as p:
            p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        ops.constraints.Plain()
        ops.numberer.RCM()
        make_system(ops)  # type: ignore[operator]
        ops.test.NormDispIncr(tol=1e-9, max_iter=10)
        ops.algorithm.Linear()
        ops.integrator.LoadControl(dlam=1.0)
        ops.analysis.Static()
        emitter = LiveOpsEmitter(wipe=True)
        ops.build().emit(emitter)
        assert emitter.analyze(steps=1) == 0
        return float(emitter.ops.nodeDisp(2, 1))

    reference = _tip(lambda o: o.system.UmfPack())
    for mt in ("unsymmetric", "symmetric", "spd"):
        got = _tip(lambda o, m=mt: o.system.Pardiso(matrix_type=m))
        assert got == pytest.approx(reference, rel=1e-12), (
            f"Pardiso matrix_type={mt!r} disagreed with UmfPack"
        )


@pytest.mark.live
def test_diagonal_drives_explicit_transient() -> None:
    """``system Diagonal`` solves a lumped-mass explicit transient."""
    ops = apeSees(cast("object", make_two_node_beam()))  # type: ignore[arg-type]
    ops.model(ndm=3, ndf=6)
    transf = ops.geomTransf.Linear(vecxz=(1.0, 0.0, 0.0))
    ops.element.elasticBeamColumn(
        pg="Cols", transf=transf,
        A=0.01, E=200e9, Iz=1e-4, Iy=1e-4, G=80e9, J=1e-4,
    )
    ops.fix(pg="Base", dofs=(1, 1, 1, 1, 1, 1))
    ops.mass(nodes=[2], values=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(node=2, forces=(1000.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    ops.constraints.Plain()
    ops.numberer.Plain()
    ops.system.Diagonal()
    ops.test.NormDispIncr(tol=1e-9, max_iter=10)
    ops.algorithm.Linear()
    # CentralDifference is in stock OpenSees, so this needs no fork build.
    ops.integrator.CentralDifference()
    ops.analysis.Transient()

    emitter = LiveOpsEmitter(wipe=True)
    ops.build().emit(emitter)
    assert emitter.analyze(steps=5, dt=1e-4) == 0
    assert emitter.ops.getTime() == pytest.approx(5e-4, rel=1e-6)

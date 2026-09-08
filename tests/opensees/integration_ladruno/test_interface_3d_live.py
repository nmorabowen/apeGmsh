"""TIMs A10 S3 — a 3-D ``g.constraints.interface()`` deck runs on the
real fork binary.

The unit and e2e files pin the deck's SHAPE (one ``zeroLength`` per
coincident pair, ``-mat mN mT mT`` on ``-dir 1 2 3``, a per-pair
six-float ``-orient``). Shape is not correctness here: fork #808 /
ADR 96 made the differing-dof and rotational-``-dir`` cases a *warning
plus an inert element* rather than a crash, so a deck apeGmsh got wrong
would still exit 0 and hand back results that are simply missing the
interface. This file is the only thing that reads the fork's own verdict.

Two boxes stacked at ``z = 1`` (soil below, footing above), meshed
node-for-node but un-fragmented, so the shared face carries two
coincident node sets — 9 pairs at ``n = 3``. An ``ENT`` normal law (no
tension whatever) and an ``epp`` Coulomb tangential law; the footing's
top face is pushed down over five ``LoadControl`` steps. Two decks:

1. **equal ndf (3, 3)** — both boxes ``stdBrick``. The vanilla pair, and
   the baseline that says the element shape itself is well-formed.
2. **mixed ndf (4, 3)** — the soil is a ``LadrunoUP`` u-p continuum, the
   footing a ``stdBrick``. This is the pair the slice exists for, and
   the only one on which the ADR 96 warnings below CAN fire. Its
   pressure datum is the base face's DOF 4 — S4 is what verifies the
   pressure is untouched by the interface; here it is only what keeps
   the static u-p solve non-singular.

Both assert that ``ops.tcl(run=True, bin=...)`` returns rather than
raising, and that the log carries no ``differing dof`` /
``passenger mode`` / ``element disabled`` line. Mutation-checked: emitting
``-dir 1 2 4`` on the mixed deck instead of ``-dir 1 2 3`` makes the fork
print "passenger mode" and disable the element, and this test fails.

Gate: ``pytest.mark.ladruno_fork`` (root ``conftest.py`` auto-skips
unless the live backend resolves to the fork) plus this file's own
``_resolve_fork_exe`` for the subprocess binary — the same two-step gate
``test_solver_stats_live.py`` uses. Measured 2026-09-08 against fork
build ``1652f945c``, which carries ADR 96.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.opensees import apeSees

pytestmark = pytest.mark.ladruno_fork

NORMAL = NormalLaw(kind="ent", k_per_area=1.0e9)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=1.0e8, tau_b=2.5e5)

#: What the fork prints when it refuses a pair or disables an element —
#: ADR 96's own wording, per ``contact_3d_passenger_dof_adoption.md``.
_FORK_REFUSALS = (
    re.compile(r"differing\s+dof", re.I),
    re.compile(r"passenger\s+mode", re.I),
    re.compile(r"element\s+disabled", re.I),
)


def _resolve_fork_exe() -> str:
    """The Tcl subprocess binary alongside the fork's ``opensees.pyd``.

    The ``ladruno_fork`` marker gates on the LIVE backend, which can
    resolve without this variable (the fork on ``PYTHONPATH`` does it
    too); the subprocess lane additionally needs the EXE, so it skips on
    its own when ``APEGMSH_OPENSEES_BIN`` is unset or does not name a
    directory holding the binary.
    """
    bin_dir = os.environ.get("APEGMSH_OPENSEES_BIN")
    if bin_dir:
        exe_name = "OpenSees.exe" if os.name == "nt" else "OpenSees"
        candidate = os.path.join(bin_dir, exe_name)
        if os.path.isfile(candidate):
            return candidate
    pytest.skip(
        "APEGMSH_OPENSEES_BIN is not set to a dist/bin containing the "
        "OpenSees Tcl binary -- TIMs A10 S3's live smoke needs the EXE "
        "for the subprocess lane, separately from the live backend the "
        "ladruno_fork marker gates on."
    )
    raise AssertionError("unreachable")  # pragma: no cover


def _surface_at_z(volume: int, z: float, tol: float = 1e-6) -> int:
    for _dim, tag in gmsh.model.getBoundary([(3, volume)], oriented=False):
        bb = gmsh.model.getBoundingBox(2, abs(tag))
        if abs(bb[2] - z) < tol and abs(bb[5] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary surface of volume {volume} at z={z}")


def _two_box_fem(*, slave_ndf: int | None = None, n: int = 3):
    """Soil ``[0,1]^3`` under footing ``[0,1]^2 x [1,2]``, un-fragmented,
    so the two surfaces at ``z = 1`` carry coincident but distinct node
    sets — the topology ``interface()`` pairs."""
    with apeGmsh(model_name="iface_a10_live", verbose=False) as g:
        soil = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        footing = g.model.geometry.add_box(0, 0, 1, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite([(3, soil), (3, footing)], n=n)
        g.mesh.generation.generate(3)
        g.physical.add(3, [soil], name="soil")
        g.physical.add(3, [footing], name="footing")
        g.physical.add(2, [_surface_at_z(soil, 1.0)], name="face")
        g.physical.add(2, [_surface_at_z(footing, 1.0)], name="skin")
        g.physical.add(2, [_surface_at_z(soil, 0.0)], name="base")
        g.physical.add(2, [_surface_at_z(footing, 2.0)], name="cap")
        g.constraints.interface(
            "face", "skin", normal=NORMAL, tangential=TANGENTIAL,
            slave_ndf=slave_ndf, name="SoilFooting")
        return g.mesh.queries.get_fem_data(dim=3)


def _analysis_chain(ops: apeSees) -> None:
    ts = ops.timeSeries.Linear()
    with ops.pattern.Plain(series=ts) as p:
        p.load(pg="cap", forces=(0.0, 0.0, -1.0e4))
    ops.constraints.Transformation()
    ops.numberer.RCM()
    ops.system.BandGeneral()
    ops.test.NormDispIncr(tol=1e-6, max_iter=50)
    ops.algorithm.Newton()
    ops.integrator.LoadControl(dlam=0.2)
    ops.analysis.Static()


def _run_and_check(ops: apeSees, deck: Path, n_pairs: int) -> None:
    """Run the deck on the fork and read its verdict.

    Returning from ``tcl(run=True)`` at all is the first half: it raises
    ``RuntimeError`` (log tail + path) on a non-zero exit or a Tcl error.
    """
    ops.tcl(str(deck), run=True, bin=_resolve_fork_exe(), analyze_steps=5)

    lines = deck.read_text().splitlines()
    assert sum(
        1 for ln in lines if ln.startswith("element zeroLength ")
    ) == n_pairs

    log = deck.with_suffix(".log")
    assert log.is_file(), f"no run log beside {deck}"
    offenders = [
        ln for ln in log.read_text(errors="replace").splitlines()
        if any(pat.search(ln) for pat in _FORK_REFUSALS)
    ]
    assert not offenders, (
        "the fork refused part of the interface — ADR 96 makes these a "
        "warning plus an inert element, so the run would otherwise pass "
        "with the springs silently absent:\n" + "\n".join(offenders)
    )


def test_3d_interface_equal_ndf_deck_runs(tmp_path: Path) -> None:
    """The vanilla ``(3, 3)`` pair: both boxes ``stdBrick``."""
    fem = _two_box_fem()
    n_pairs = len(fem.elements.interfaces)
    assert n_pairs == 9

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("soil", "footing"):
        ops.element.stdBrick(pg=pg, material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    _analysis_chain(ops)

    _run_and_check(ops, tmp_path / "iface3d_equal.tcl", n_pairs)


def test_3d_interface_up_soil_deck_runs(tmp_path: Path) -> None:
    """The ``(4, 3)`` pair PM-01 actually has: an ndf-4 u-p soil under an
    ndf-3 footing skin. Fork #808 / ADR 96 joins it directly, acting on
    DOFs 1-3 with the pore pressure an untouched passenger — and says so
    in the log if it cannot."""
    fem = _two_box_fem(slave_ndf=3)
    n_pairs = len(fem.elements.interfaces)
    assert n_pairs == 9

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    soil_mat = ops.nDMaterial.ElasticIsotropic(E=3e7, nu=0.3, rho=2.0)
    ops.element.LadrunoUP(
        pg="soil", material=soil_mat,
        Kf=2.2e6, poro=0.4, rhoF=1.0, perm=(1e-4,) * 3,
    )
    footing_mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.stdBrick(pg="footing", material=footing_mat)
    # DOF 4 on the base is the u-p pressure datum, not an interface
    # concern: a sealed static u-p region is singular in p and every
    # serial solver factorises it through round-off (ADR 0074 gate).
    ops.fix(pg="base", dofs=(1, 1, 1, 1))
    _analysis_chain(ops)

    _run_and_check(ops, tmp_path / "iface3d_up.tcl", n_pairs)

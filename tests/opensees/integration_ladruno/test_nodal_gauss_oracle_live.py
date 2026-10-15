"""Fork lane: nodal and Gauss value oracles on MPCO and ``.ladruno``.

Gap G16 of the DomainCapture parity table (#1767, #1945). The MPCO proofs
in ``tests/test_results_mpco_emit_real.py`` and
``tests/test_results_mpco_element_real.py`` import ``openseespy`` by name and
need an MPCO-capable build or a Tcl launcher, so no CI lane runs them: pip
``openseespy`` 3.8.0 has no MPCO recorder. This file runs the same oracles in
process on the module the bridge drives (``get_ops()``, the fork on the
``live-fork`` lane), on both of the fork's HDF5 formats:

* **Nodes:** every recorded node and component of ``displacement`` equals
  ``ops.nodeDisp(n)`` captured at the same step, and ``reaction_force``
  equals ``ops.nodeReaction(n)`` after ``ops.reactions()``.
* **Gauss:** every GP and component of ``stresses`` equals
  ``ops.eleResponse(e, "stresses")`` reshaped ``(n_gp, 6)``, for a
  ``FourNodeTetrahedron`` (1 GP) and an 8-node ``stdBrick`` (8 GPs, so the GP
  order is under test).

Both at ``rtol=1e-10``, over two load steps. The last test perturbs one
recorded value in a copy of each file and shows the oracle fails, so a
reader that drops, reorders or rescales a value cannot pass.

Node and element tags are not 1..n, so an id lookup that assumes them fails.

Gated by the ``ladruno_fork`` marker (root conftest auto-skips off-fork).
"""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest

from apeGmsh.opensees.emitter.live import get_ops
from apeGmsh.results import Results

from tests.conftest import _stub_model_h5_path

pytestmark = pytest.mark.ladruno_fork

RTOL = 1e-10
N_STEPS = 2

DISP = ("displacement_x", "displacement_y", "displacement_z")
REAC = ("reaction_force_x", "reaction_force_y", "reaction_force_z")
#: ``eleResponse(e, "stresses")`` order at each GP. Spelled out, not read from
#: RESPONSE_CATALOG: the reader under test takes its layout from the catalog,
#: so deriving the oracle from it would assert nothing.
STRESS_3D = ("stress_xx", "stress_yy", "stress_zz",
             "stress_xy", "stress_yz", "stress_xz")

#: One solid per family, under a load with every component non-trivial.
#: The tet's base is tilted: on a flat base the apex shape function's
#: gradient is (0, 0, 1) and the strain has no xy term at all.
MODELS: dict[str, dict[str, Any]] = {
    "tet": dict(
        cls="FourNodeTetrahedron", tag=7, n_gp=1,
        nodes={11: (0.0, 0.0, 0.0), 12: (1.0, 0.0, 0.2),
               13: (0.0, 1.0, 0.4), 14: (0.3, 0.4, 1.0)},
        fixed=(11, 12, 13),
        loads={14: (2.0e5, -1.0e5, -1.0e6)},
    ),
    "brick": dict(
        cls="stdBrick", tag=9, n_gp=8,
        nodes={21: (0.0, 0.0, 0.0), 22: (1.0, 0.0, 0.0),
               23: (1.0, 1.0, 0.0), 24: (0.0, 1.0, 0.0),
               25: (0.0, 0.0, 1.0), 26: (1.0, 0.0, 1.0),
               27: (1.0, 1.0, 1.0), 28: (0.0, 1.0, 1.0)},
        fixed=(21, 22, 23, 24),
        loads={25: (1.0e5, 0.0, -1.0e6), 26: (0.0, 2.0e5, -1.0e6),
               27: (-1.0e5, 0.0, -2.0e6), 28: (3.0e5, 1.0e5, -5.0e5)},
    ),
}

#: ``stdBrick``'s GP order, from the fork's Brick.cpp (the ``i, j, k`` loop at
#: L535-541 over ``sg = {-1/sqrt(3), +1/sqrt(3)}``): zeta fastest, GP index
#: ``i*4 + j*2 + k``. Spelled out, not read from RESPONSE_CATALOG.
_G = 1.0 / np.sqrt(3.0)
BRICK_GP = np.array([[a, b, c] for a in (-_G, _G) for b in (-_G, _G)
                     for c in (-_G, _G)])

KINDS = ("mpco", "ladruno")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _solve(kind: str, family: str, path: Path) -> list[dict[str, Any]]:
    """Run ``family`` on the fork, recording to ``path``; return the live
    domain's answer at every step."""
    m = MODELS[family]
    ops = get_ops()
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 3)
    for nid, xyz in m["nodes"].items():
        ops.node(nid, *xyz)
    for nid in m["fixed"]:
        ops.fix(nid, 1, 1, 1)
    ops.nDMaterial("ElasticIsotropic", 1, 200e9, 0.3)
    ops.element(m["cls"], m["tag"], *m["nodes"], 1)
    ops.recorder(kind, str(path), "-N", "displacement", "reactionForce",
                 "-E", "stresses")
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid, f in m["loads"].items():
        ops.load(nid, *f)
    ops.system("ProfileSPD")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0 / N_STEPS)
    ops.analysis("Static")
    live = []
    for _ in range(N_STEPS):
        assert ops.analyze(1) == 0
        ops.reactions()
        live.append(dict(
            disp={n: np.asarray(ops.nodeDisp(n), dtype=float)
                  for n in m["nodes"]},
            reac={n: np.asarray(ops.nodeReaction(n), dtype=float)
                  for n in m["nodes"]},
            stress=np.asarray(ops.eleResponse(m["tag"], "stresses"),
                              dtype=float).reshape(m["n_gp"], 6),
        ))
    ops.remove("recorders")      # flush and close the file
    ops.wipe()
    assert path.exists(), path
    # The load is shared out over every component, so a zero read back
    # cannot pass by accident.
    last = live[-1]
    assert np.all(last["stress"] != 0.0)
    return live


def _open(kind: str, path: Path) -> Results:
    if kind == "mpco":
        return Results.from_mpco(str(path), model_h5=_stub_model_h5_path())
    return Results.from_ladruno(str(path))


def _close(want: np.ndarray, got: np.ndarray, msg: str) -> None:
    np.testing.assert_allclose(
        got, want, rtol=RTOL, atol=RTOL * float(np.abs(want).max()),
        err_msg=msg,
    )


def _assert_nodal(r: Results, family: str, live: list[dict[str, Any]]) -> None:
    nodes = list(MODELS[family]["nodes"])
    (stage,) = r.stages
    assert stage.n_steps == N_STEPS
    s = r.stage(stage.id)
    for channel, comps in (("disp", DISP), ("reac", REAC)):
        for k, comp in enumerate(comps):
            slab = s.nodes.get(component=comp)
            ids = [int(n) for n in slab.node_ids]
            assert sorted(ids) == sorted(nodes), comp
            got = np.asarray(slab.values)
            assert got.shape == (N_STEPS, len(nodes)), (comp, got.shape)
            want = np.array([[step[channel][n][k] for n in ids]
                             for step in live])
            _close(want, got, f"{comp} on {family}")


def _assert_gauss(r: Results, family: str, live: list[dict[str, Any]]) -> None:
    m = MODELS[family]
    s = r.stage(r.stages[0].id)
    for k, comp in enumerate(STRESS_3D):
        slab = s.elements.gauss.get(component=comp)
        assert [int(e) for e in slab.element_index] == [m["tag"]] * m["n_gp"]
        got = np.asarray(slab.values)
        assert got.shape == (N_STEPS, m["n_gp"]), (comp, got.shape)
        want = np.array([step["stress"][:, k] for step in live])
        _close(want, got, f"{comp} on {family}")
    xi = np.asarray(slab.natural_coords)
    if family == "tet":
        # The one-point rule sits at the centroid.
        np.testing.assert_allclose(xi, [[0.25, 0.25, 0.25]], atol=1e-12)
    else:
        # GP order, not just the set: the value columns above follow it.
        np.testing.assert_allclose(xi, BRICK_GP, atol=1e-12)


# ---------------------------------------------------------------------------
# Value oracles
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", sorted(MODELS))
@pytest.mark.parametrize("kind", KINDS)
def test_nodal_values_match_live_domain(
    tmp_path: Path, kind: str, family: str,
) -> None:
    path = tmp_path / f"{family}.{kind}"
    live = _solve(kind, family, path)
    with _open(kind, path) as r:
        _assert_nodal(r, family, live)


@pytest.mark.parametrize("family", sorted(MODELS))
@pytest.mark.parametrize("kind", KINDS)
def test_gauss_values_match_live_domain(
    tmp_path: Path, kind: str, family: str,
) -> None:
    path = tmp_path / f"{family}.{kind}"
    live = _solve(kind, family, path)
    with _open(kind, path) as r:
        _assert_gauss(r, family, live)


# ---------------------------------------------------------------------------
# Mutation check: the oracle sees one perturbed value
# ---------------------------------------------------------------------------


def _perturb(path: Path, kind: str, group: str) -> None:
    """Scale one entry of the last step of ``group`` by ``1 + 1e-6``."""
    base = "MODEL_STAGE[1]/RESULTS/"
    with h5py.File(path, "r+") as f:
        res = f[base + ("ON_ELEMENTS/stresses" if group == "stresses"
                        else f"ON_NODES/{group}")]
        if group == "stresses":
            (bucket,) = list(res)
            res = res[bucket]
        data = res["DATA"]
        # MPCO keeps one dataset per step. Its STEP_<k> names count on
        # across wipes in one process, so take the last by its STEP attr.
        ds = (data[max(data, key=lambda k: int(data[k].attrs["STEP"][0]))]
              if kind == "mpco" else data)
        arr = ds[...]
        view = arr[-1] if kind == "ladruno" else arr
        # The largest entry, so the change is far above rtol and atol.
        idx = np.unravel_index(np.abs(view).argmax(), view.shape)
        view[idx] *= 1.0 + 1e-6
        ds[...] = arr


@pytest.mark.parametrize("group", ["DISPLACEMENT", "REACTION_FORCE", "stresses"])
@pytest.mark.parametrize("kind", KINDS)
def test_oracle_rejects_a_perturbed_record(
    tmp_path: Path, kind: str, group: str,
) -> None:
    path = tmp_path / f"brick.{kind}"
    live = _solve(kind, "brick", path)
    check = _assert_gauss if group == "stresses" else _assert_nodal
    with _open(kind, path) as r:
        check(r, "brick", live)               # the clean file passes
    bad = tmp_path / f"bad.{kind}"
    shutil.copyfile(path, bad)
    _perturb(bad, kind, group)
    with _open(kind, bad) as r, pytest.raises(AssertionError):
        check(r, "brick", live)

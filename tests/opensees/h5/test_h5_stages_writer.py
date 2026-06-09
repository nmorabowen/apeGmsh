"""ADR 0055 Phase 2 — ``/opensees/stages`` writer (schema 2.18.0).

Group-shape coverage for the staged-H5 capture pipeline: the H5
emitter routes the per-stage emit stream into ``_StageEmitBlock``
buckets (in-band capture inside the ``stage_open`` … ``stage_close``
bracket) and ``set_stage_records`` attaches the declarative complement
(``activated_pgs``, per-stage ``initial_stress``,
``activate_absorbing``).  ``test_h5_staged_fail_loud.py`` pins the
guard contract (partitioned raises, read side raises); this module
pins WHAT the writer persists:

* owned topology (activation → ``owned_node_ids`` /
  ``owned_element_ids`` in emit order),
* stage-bound BCs with the global-ndf compound width,
* per-stage analysis chain scoped to the stage group (no global leak),
* stage patterns (resolved loads),
* presence-encoded tri-state (``set_time`` / ``analyze_dt`` absent
  unless the stage set them),
* per-stage declarative initial stress,
* structural determinism across two writes of the same build.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from apeGmsh.opensees.apesees import apeSees

from tests.opensees.fixtures.fem_stub import (
    FEMStub,
    _ElementGroupView,
    _ElementsStub,
    _NodesStub,
)


# ---------------------------------------------------------------------------
# Fixture — two-quad stub: "Rock" stays global, "Fill" activates in a stage
# ---------------------------------------------------------------------------


def _make_two_quad_fem_stub() -> FEMStub:
    return FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4, 5, 6],
            coords=[
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 2.0, 0.0),
                (0.0, 2.0, 0.0),
            ],
            node_pgs={
                "Rock": [1, 2, 3, 4],
                "Fill": [3, 4, 5, 6],
                "Base": [1, 2],
                "FillTop": [5, 6],
            },
        ),
        elements=_ElementsStub(
            elem_pgs={
                "Rock": _ElementGroupView(
                    ids=(1,), connectivity=((1, 2, 3, 4),),
                ),
                "Fill": _ElementGroupView(
                    ids=(2,), connectivity=((4, 3, 5, 6),),
                ),
            },
        ),
    )


def _chain(ops: apeSees) -> dict[str, object]:
    return {
        "test":        ops.test.NormDispIncr(tol=1e-4, max_iter=50),
        "algorithm":   ops.algorithm.Newton(),
        "integrator":  ops.integrator.LoadControl(dlam=0.1),
        "constraints": ops.constraints.Plain(),
        "numberer":    ops.numberer.RCM(),
        "system":      ops.system.UmfPack(),
        "analysis":    ops.analysis.Static(),
    }


def _build_two_stage_bridge() -> apeSees:
    """Global "Rock" quad + global fix; stage 1 activates "Fill" with a
    stage fix; stage 2 sets time, loads through a stage pattern, and
    ramps a per-stage initial stress."""
    fem = _make_two_quad_fem_stub()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    with ops.stage(name="construction") as s:
        s.activate(pgs=["Fill"])
        s.fix(pg="FillTop", dofs=(1, 1))
        s.analysis(**_chain(ops))
        s.run(n_increments=5)  # static — no dt (tri-state: attr absent)

    with ops.stage(name="loading") as s:
        s.set_time(2.5)
        ts = ops.timeSeries.Linear()
        with s.pattern(series=ts) as p:
            p.load(pg="Fill", forces=(10.0, 0.0))
        s.initial_stress(
            name="insitu", pg="Rock",
            sigma_xx=-1.0e3, sigma_yy=-1.0e3, sigma_zz=-2.0e3,
            ramp_steps=4,
        )
        s.analysis(**_chain(ops))
        s.run(n_increments=3, dt=0.01)
    return ops


# ---------------------------------------------------------------------------
# Group shape
# ---------------------------------------------------------------------------


def test_stages_group_shape(tmp_path: Path) -> None:
    out = tmp_path / "staged.h5"
    _build_two_stage_bridge().h5(str(out))

    with h5py.File(str(out), "r") as f:
        ops_grp = f["opensees"]
        # The stage chains are scoped per stage — no global analysis
        # group on a staged file (phantom-leak regression guard).
        assert "analysis" not in ops_grp
        stages = ops_grp["stages"]
        assert int(stages.attrs["n_stages"]) == 2
        assert sorted(stages.keys()) == ["stage_000", "stage_001"]

        # -- stage_000: activation + stage fix, static analyze --------
        g0 = stages["stage_000"]
        assert g0.attrs["name"] == "construction"
        assert int(g0.attrs["analyze_steps"]) == 5
        assert "analyze_dt" not in g0.attrs       # dt never set
        assert "set_time" not in g0.attrs         # never called
        assert "set_creep_on" not in g0.attrs
        assert "pre_analyze_reset" not in g0.attrs
        assert int(g0.attrs["domain_change"]) == 1
        assert [s.decode() if isinstance(s, bytes) else s
                for s in g0["activated_pgs"][()]] == ["Fill"]
        # Owned topology: nodes 5, 6 are referenced ONLY by the Fill
        # quad (3, 4 are shared with the global Rock quad → global).
        assert sorted(int(n) for n in g0["owned_node_ids"][()]) == [5, 6]
        assert len(g0["owned_element_ids"][()]) == 1
        # Stage-bound fix — compound width == global ndf envelope.
        fix_rows = g0["bcs"]["fix"][()]
        assert len(fix_rows) == 2  # "Base" fan-out: nodes 1, 2
        assert fix_rows.dtype["dofs"].shape == (2,)
        # Per-stage chain, scoped to the stage group.
        chain = g0["analysis"]
        assert chain.attrs["algorithm"] == "Newton"
        assert chain.attrs["numberer"] == "RCM"
        assert chain.attrs["analysis"] == "Static"
        # No pattern / initial stress in this stage.
        assert "patterns" not in g0
        assert "initial_stress" not in g0

        # -- stage_001: set_time + pattern + initial stress, transient -
        g1 = stages["stage_001"]
        assert g1.attrs["name"] == "loading"
        assert int(g1.attrs["analyze_steps"]) == 3
        assert float(g1.attrs["analyze_dt"]) == 0.01
        assert float(g1.attrs["set_time"]) == 2.5
        # No topology / BC mutation in this stage → no domainChange.
        assert int(g1.attrs["domain_change"]) == 0
        assert "activated_pgs" not in g1
        assert "owned_node_ids" not in g1
        assert "bcs" not in g1
        # Stage pattern with resolved loads (Fill fan-out: 4 nodes).
        pats = g1["patterns"]
        (pat_name,) = list(pats.keys())
        loads = pats[pat_name]["loads"][()]
        assert len(loads) == 4
        # Per-stage declarative initial stress (Phase-1 field set).
        s0 = g1["initial_stress"]["stress_000"]
        assert s0.attrs["name"] == "insitu"
        assert float(s0.attrs["sigma_zz"]) == -2.0e3
        assert int(s0.attrs["ramp_steps"]) == 4
        assert s0.attrs["pg"] == "Rock"
        assert "elements" not in s0


def test_global_zone_untouched_by_stage_records(tmp_path: Path) -> None:
    """Stage-bound fixes must NOT leak into the global ``/opensees/bcs``
    (they'd double-apply as t=0 BCs on replay), while element metadata
    stays complete (dual-append: the reader rebuilds the full pool
    from ``element_meta``, staged elements included)."""
    out = tmp_path / "staged.h5"
    _build_two_stage_bridge().h5(str(out))

    with h5py.File(str(out), "r") as f:
        ops_grp = f["opensees"]
        # Global fix: exactly the 2 pre-stage "Base" rows; the stage's
        # own 2 fix rows live under stage_000/bcs.
        assert len(ops_grp["bcs"]["fix"][()]) == 2
        # element_meta carries BOTH quads (global + stage-owned).
        quad_meta = ops_grp["element_meta"]["quad"]
        assert len(quad_meta["ids"][()]) == 2
        # fem_eids column present → the fem_eid→ops_tag map is already
        # persisted; no separate element_tag_map dataset exists.
        assert "fem_eids" in quad_meta
        assert "element_tag_map" not in ops_grp


# ---------------------------------------------------------------------------
# Determinism — same build, two writes, identical stages zone
# ---------------------------------------------------------------------------


def _norm(v: Any) -> Any:
    """Recursively normalize h5py/numpy values to plain python."""
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return _norm(v.item())
    if isinstance(v, np.ndarray):
        return [_norm(x) for x in v.tolist()]
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _collect_zone(g: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def visit(name: str, obj: Any) -> None:
        attrs = {k: _norm(obj.attrs[k]) for k in sorted(obj.attrs)}
        if isinstance(obj, h5py.Dataset):
            out[name] = ["dataset", str(obj.dtype), _norm(obj[()]), attrs]
        else:
            out[name] = ["group", attrs]

    g.visititems(visit)
    return out


def test_stages_zone_deterministic_across_writes(tmp_path: Path) -> None:
    ops = _build_two_stage_bridge()
    out_a = tmp_path / "a.h5"
    out_b = tmp_path / "b.h5"
    ops.h5(str(out_a))
    ops.h5(str(out_b))

    with h5py.File(str(out_a), "r") as fa, h5py.File(str(out_b), "r") as fb:
        za = _collect_zone(fa["opensees"]["stages"])
        zb = _collect_zone(fb["opensees"]["stages"])

    assert za == zb

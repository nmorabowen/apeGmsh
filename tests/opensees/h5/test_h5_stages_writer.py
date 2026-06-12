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
        # domainChange is an unconditional stage barrier (recorder
        # MODEL_STAGE boundaries key off the domain-change stamp) —
        # even a pure-loading stage records it.
        assert int(g1.attrs["domain_change"]) == 1
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
    """Recursively normalize h5py/numpy values to plain python.

    NaN sentinels (the ``_write_param_array`` string-slot markers)
    normalize to the literal string ``"NaN"`` so equality comparison
    works (``nan != nan`` would make identical zones compare unequal).
    """
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.generic):
        return _norm(v.item())
    if isinstance(v, float) and v != v:
        return "NaN"
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


# ---------------------------------------------------------------------------
# Kitchen-sink stage — the writer branches the 2-stage fixture leaves dead
# (gate-2: support/sp_holds, mass, removals, rayleigh + scoped attach,
# recorder, set_creep, reset, activate_absorbing)
# ---------------------------------------------------------------------------


def _build_kitchen_sink_bridge() -> apeSees:
    fem = _make_two_quad_fem_stub()
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    ops.element.FourNodeQuad(pg="Fill", thickness=1.0, material=mat)
    ops.fix(pg="Base", dofs=(1, 1))

    with ops.stage(name="mutate") as s:
        s.activate(pgs=["Fill"])
        s.support(pg="FillTop", dofs=(1, 0))
        s.mass(pg="FillTop", values=(2.0, 2.0))
        s.remove_sp(pg="Base", dofs=(1,))
        s.remove_element(elements=[1])  # the global Rock quad (fem eid)
        s.damping.rayleigh(alpha_m=0.05)              # global form
        s.damping.rayleigh(alpha_m=0.1, on="Fill")    # region-scoped
        s.recorder(ops.recorder.Node(
            file="r.out", response="disp", nodes=(5,), dofs=(1, 2),
        ))
        s.set_creep(True)
        s.reset()
        s.activate_absorbing(pg="Fill")
        s.analysis(**_chain(ops))
        s.run(n_increments=2, dt=0.5)
    return ops


def test_kitchen_sink_stage_branches(tmp_path: Path) -> None:
    out = tmp_path / "sink.h5"
    _build_kitchen_sink_bridge().h5(str(out))

    with h5py.File(str(out), "r") as f:
        g = f["opensees"]["stages"]["stage_000"]
        # Presence-encoded mutators.
        assert int(g.attrs["set_creep_on"]) == 1
        assert int(g.attrs["pre_analyze_reset"]) == 1
        # Removals (resolved targets; single-dof reshape hazard).
        rsp = g["remove_sp"][()]
        assert rsp.shape == (2, 2)          # Base nodes 1, 2 × dof 1
        assert sorted(int(r[0]) for r in rsp) == [1, 2]
        assert {int(r[1]) for r in rsp} == {1}
        rel = g["remove_element"][()]
        assert len(rel) == 1                # the Rock quad's ops tag
        # Stage mass (FillTop fan-out: nodes 5, 6).
        mass_rows = g["bcs"]["mass"][()]
        assert len(mass_rows) == 2
        # HOLD support pattern: role attr + sp_holds rows + emit_index.
        pats = g["patterns"]
        hold_names = [
            n for n in pats if pats[n].attrs.get("role") == "hold"
        ]
        assert len(hold_names) == 1
        hold = pats[hold_names[0]]
        sp_holds = hold["sp_holds"][()]
        assert sorted((int(r[0]), int(r[1])) for r in sp_holds) == [
            (5, 1), (6, 1),
        ]
        assert int(hold.attrs["emit_index"]) > 0
        # Stage rayleigh: global form dataset + emit-order stamps; the
        # region-scoped form rides the regions echo with kind=rayleigh.
        ray = g["rayleigh"][()]
        assert ray.shape == (1, 4)
        assert float(ray[0][0]) == 0.05
        ray_seq = g["rayleigh_emit_index"][()]
        assert len(ray_seq) == 1
        regions = g["regions"]
        kinds = {
            regions[n].attrs["kind"] for n in regions
        }
        assert "rayleigh" in kinds
        # Relative order preserved: the global rayleigh call emits
        # before its region-scoped sibling (slot 5b internal order).
        scoped = [
            int(regions[n].attrs["emit_index"])
            for n in regions
            if regions[n].attrs["kind"] == "rayleigh"
        ]
        assert all(int(ray_seq[0]) < s for s in scoped)
        # Stage-bound recorder.
        assert len(list(g["recorders"])) == 1
        # Absorbing flip — declarative pg.
        ab = g["activate_absorbing"]["absorb_000"]
        assert ab.attrs["pg"] == "Fill"
        assert "elements" not in ab


# ---------------------------------------------------------------------------
# Direct-emitter guards (gate-2: the emit-then-write() bypass) + MP capture
# ---------------------------------------------------------------------------


def _drive_minimal_stage(e: Any) -> None:
    e.model(ndm=2, ndf=2)
    e.stage_open("s")
    e.equalDOF(1, 2, 1, 2)
    e.analyze(steps=1)
    e.stage_close()


def test_direct_write_without_set_stage_records_raises(
    tmp_path: Path,
) -> None:
    from apeGmsh.opensees.emitter.h5 import H5Emitter
    import pytest

    e = H5Emitter()
    _drive_minimal_stage(e)
    with pytest.raises(RuntimeError, match="set_stage_records"):
        e.write(str(tmp_path / "bypass.h5"))


def test_write_with_open_stage_bracket_raises(tmp_path: Path) -> None:
    from apeGmsh.opensees.emitter.h5 import H5Emitter
    import pytest

    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    e.stage_open("dangling")
    with pytest.raises(RuntimeError, match="still open"):
        e.write(str(tmp_path / "open.h5"))


def test_set_stage_records_mp_constraint_roundtrips_to_stage_group(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    _drive_minimal_stage(e)
    rec = SimpleNamespace(
        name="s", n_increments=1, dt=None, activated_pgs=(),
        initial_stress_records=(), activate_absorbing_records=(),
    )
    e.set_stage_records([rec])
    out = tmp_path / "mp.h5"
    e.write(str(out))
    with h5py.File(str(out), "r") as f:
        rows = f["opensees"]["stages"]["stage_000"]["constraints"][
            "equalDOF"
        ][()]
        assert len(rows) == 1
        assert int(rows[0]["master"]) == 1
        assert int(rows[0]["slave"]) == 2
        # Global constraints zone untouched by the stage rows.
        assert "constraints" not in f["opensees"] or "equalDOF" not in f[
            "opensees"
        ]["constraints"]


def test_set_stage_records_dt_mismatch_raises(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import pytest

    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    _drive_minimal_stage(e)   # captured dt=None
    rec = SimpleNamespace(
        name="s", n_increments=1, dt=0.1, activated_pgs=(),
        initial_stress_records=(), activate_absorbing_records=(),
    )
    with pytest.raises(RuntimeError, match="analyze capture mismatch"):
        e.set_stage_records([rec])


def test_stage_phantom_node_fails_loud(tmp_path: Path) -> None:
    from types import SimpleNamespace

    import pytest

    from apeGmsh.opensees._internal.tag_resolution import (
        set_phantom_node_tags,
    )
    from apeGmsh.opensees.emitter.h5 import H5Emitter

    e = H5Emitter()
    e.model(ndm=2, ndf=2)
    set_phantom_node_tags(e, {99})
    e.stage_open("s")
    e.node(99, 0.0, 0.0, 0.0, ndf=6)   # stage-emitted phantom
    e.analyze(steps=1)
    e.stage_close()
    rec = SimpleNamespace(
        name="s", n_increments=1, dt=None, activated_pgs=(),
        initial_stress_records=(), activate_absorbing_records=(),
    )
    with pytest.raises(NotImplementedError, match="phantom"):
        e.set_stage_records([rec])


# ---------------------------------------------------------------------------
# model_hash (gate-2: the commit's central hash claim was untested)
# ---------------------------------------------------------------------------


def _model_hash_of(path: Path) -> str:
    """Read the stamped model_hash via h5py — the staged read probe
    makes ``OpenSeesModel.from_h5(...).lineage`` unreachable."""
    with h5py.File(str(path), "r") as f:
        return str(f["meta"]["lineage"].attrs["model_hash"])


def test_stages_fold_into_model_hash(tmp_path: Path) -> None:
    """Two-stage vs one-stage-removed must hash differently — the
    stages zone is authored state, not a regenerable carve-out."""
    two = tmp_path / "two.h5"
    one = tmp_path / "one.h5"
    _build_two_stage_bridge().h5(str(two))

    # Same model, first stage only.
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
        s.run(n_increments=5)
    ops.h5(str(one))

    assert _model_hash_of(two) != _model_hash_of(one)


def test_two_writes_same_model_hash(tmp_path: Path) -> None:
    ops = _build_two_stage_bridge()
    a = tmp_path / "a.h5"
    b = tmp_path / "b.h5"
    ops.h5(str(a))
    ops.h5(str(b))
    assert _model_hash_of(a) == _model_hash_of(b)


# ---------------------------------------------------------------------------
# Exactly-one-partition staged build flows the flat capture path
# (is_partitioned is len > 1; the guard boundary must match dispatch)
# ---------------------------------------------------------------------------


def test_one_partition_staged_build_writes(tmp_path: Path) -> None:
    fem = _make_two_quad_fem_stub()
    fem.set_partitions([(0, [1, 2, 3, 4, 5, 6], [1, 2])])
    ops = apeSees(fem, default_orientation=None)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=1e6, nu=0.3, rho=0.0)
    ops.element.FourNodeQuad(pg="Rock", thickness=1.0, material=mat)
    with ops.stage(name="only") as s:
        s.analysis(**_chain(ops))
        s.run(n_increments=1)
    out = tmp_path / "one_part.h5"
    ops.h5(str(out))
    with h5py.File(str(out), "r") as f:
        assert "stages" in f["opensees"]

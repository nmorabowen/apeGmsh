"""The shell stress resultants are reachable from a declarative spec.

ADR 0098 A6.6 R2. ``_vocabulary.SHELL_STRESS_RESULTANTS`` named all
eight and ``_domain._gauss_record_tokens`` was written for them, but
they were never wired into ``ALL_CANONICAL`` — so
``spec.gauss(components=["membrane_force_xx", ...])`` was accepted at
declaration and died at resolve with ``Unknown component
'membrane_force_xx'``. Shells could not be contoured at all.

Two halves, both gated here:

* the vocabulary half — the eight resultants and their eight
  conjugate generalized strains are canonical, carry a shorthand, and
  belong to the ``gauss`` category;
* the honesty half — the upstream shells (``ShellMITC4`` &c.) answer
  ``ops.eleResponse(eid, "stresses")`` with a correctly sized vector of
  ZEROS, so making the vocabulary reachable would have made a flat
  contour the default shell picture. Those classes are skipped with a
  reason instead.

The live-ops cases carry the load: a unit test that only asks
``is_canonical`` would pass against a vocabulary wired to nothing.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from apeGmsh._vocabulary import (
    SHELL_GENERALIZED_STRAINS,
    SHELL_STRESS_RESULTANTS,
    expand_shorthand,
    is_canonical,
)
from apeGmsh.results.capture.spec import DomainCaptureSpec

ops = pytest.importorskip("openseespy.opensees")

from tests.conftest import _open_model_from_h5  # noqa: E402


# =====================================================================
# Vocabulary — the declaration boundary
# =====================================================================

@pytest.mark.parametrize(
    "name", SHELL_STRESS_RESULTANTS + SHELL_GENERALIZED_STRAINS,
)
def test_shell_component_is_canonical(name: str) -> None:
    assert is_canonical(name), (
        f"{name!r} is named in _vocabulary but absent from "
        f"ALL_CANONICAL, so expand_shorthand() rejects it at resolve."
    )


def test_shell_shorthands_do_not_clip_with_ndm() -> None:
    """The eight are one response layout, not a tensor.

    ``ndm``/``ndf`` clipping is for genuine vectors and tensors. A
    shell's resultants come back eight-at-a-time from one
    ``eleResponse`` call whatever the model dimension, exactly like the
    ``section_force`` line-station pair, so clipping would drop
    components the element really emits.
    """
    assert expand_shorthand("shell_resultant", ndm=2, ndf=3) == (
        SHELL_STRESS_RESULTANTS
    )
    assert expand_shorthand("shell_deformation", ndm=3, ndf=6) == (
        SHELL_GENERALIZED_STRAINS
    )


@pytest.mark.parametrize(
    "name", SHELL_STRESS_RESULTANTS + SHELL_GENERALIZED_STRAINS,
)
def test_shell_component_belongs_to_gauss(name: str) -> None:
    assert DomainCaptureSpec.where_does(name) == ("gauss",)


def test_gauss_shorthands_advertise_the_shell_pair() -> None:
    """What ``shorthands_for`` offers is what the spec will accept."""
    offered = DomainCaptureSpec.shorthands_for("gauss")
    assert offered.get("shell_resultant") == SHELL_STRESS_RESULTANTS
    assert offered.get("shell_deformation") == SHELL_GENERALIZED_STRAINS


def test_resultants_and_generalized_strains_cannot_share_a_record() -> None:
    """They are work-conjugates under different ``eleResponse`` keywords."""
    from apeGmsh.results.capture._domain import _gauss_record_tokens
    from apeGmsh.results.capture.spec import ResolvedDomainCaptureRecord

    ok = ResolvedDomainCaptureRecord(
        category="gauss", name="r", components=SHELL_STRESS_RESULTANTS,
        dt=None, n_steps=None, element_ids=np.array([1], dtype=np.int64),
    )
    assert _gauss_record_tokens(ok) == ("stress", "stresses")

    mixed = ResolvedDomainCaptureRecord(
        category="gauss", name="r",
        components=("membrane_force_xx", "curvature_xx"),
        dt=None, n_steps=None, element_ids=np.array([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="work-conjugate"):
        _gauss_record_tokens(mixed)


# =====================================================================
# Live ops — the end-to-end path the declaration boundary guards
# =====================================================================

class _MinimalFem:
    """Only what DomainCapture reads: node ids/coords and a hash."""

    def __init__(self, node_ids: np.ndarray, coords: np.ndarray) -> None:
        self.nodes = SimpleNamespace(ids=node_ids, coords=coords)
        self.elements = []

    @property
    def snapshot_id(self) -> str:
        from apeGmsh.mesh._femdata_hash import compute_snapshot_id
        return compute_snapshot_id(self)

    def to_native_h5(self, group) -> None:
        from apeGmsh.mesh._femdata_h5_io import write_neutral_zone_into_group
        write_neutral_zone_into_group(self, group, ndf=6)


_SHELL_COORDS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
], dtype=np.float64)

# A free-standing brick, 10 m away so it shares no node with the shell.
_BRICK_COORDS = np.array([
    [10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [11.0, 1.0, 0.0], [10.0, 1.0, 0.0],
    [10.0, 0.0, 1.0], [11.0, 0.0, 1.0], [11.0, 1.0, 1.0], [10.0, 1.0, 1.0],
], dtype=np.float64)

_SHELL_EID = 1
_BRICK_EID = 2


def _build_model(shell_class: str, *, with_brick: bool) -> _MinimalFem:
    """One loaded shell (+ optionally one loaded brick) in one domain."""
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)

    for nid, xyz in enumerate(_SHELL_COORDS, start=1):
        ops.node(nid, *(float(v) for v in xyz))
    for nid in (1, 4):
        ops.fix(nid, 1, 1, 1, 1, 1, 1)
    ops.section("ElasticMembranePlateSection", 1, 2.0e10, 0.2, 0.2, 2400.0)
    ops.element(shell_class, _SHELL_EID, 1, 2, 3, 4, 1)

    node_ids = list(range(1, 5))
    coords = _SHELL_COORDS
    if with_brick:
        for k, xyz in enumerate(_BRICK_COORDS, start=11):
            ops.node(k, *(float(v) for v in xyz), "-ndf", 3)
        for k in (11, 12, 13, 14):
            ops.fix(k, 1, 1, 1)
        ops.nDMaterial("ElasticIsotropic", 2, 2.0e7, 0.25)
        ops.element("stdBrick", _BRICK_EID, *range(11, 19), 2)
        node_ids += list(range(11, 19))
        coords = np.vstack([coords, _BRICK_COORDS])

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in (2, 3):
        ops.load(nid, 0.0, 0.0, -1.0e4, 0.0, 0.0, 0.0)
    if with_brick:
        for k in (15, 16, 17, 18):
            ops.load(k, 0.0, 0.0, -1.0e3)

    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1e-8, 10)
    ops.algorithm("Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    assert ops.analyze(1) == 0
    return _MinimalFem(np.array(node_ids, dtype=np.int64), coords)


def _capture(spec: DomainCaptureSpec, fem: _MinimalFem, path: Path) -> None:
    from apeGmsh.results.capture._domain import DomainCapture

    resolved = spec._resolve_with_explicit_ndm_ndf(fem, ndm=3, ndf=6)
    with DomainCapture(resolved, path, fem) as cap:
        cap.begin_stage("static_load", kind="static")
        cap.step(t=ops.getTime())
        cap.end_stage()


def _gauss_level():
    from apeGmsh.results.readers._protocol import ResultLevel
    return ResultLevel.GAUSS


def test_declared_shell_record_captures_real_resultants(
    tmp_path: Path,
) -> None:
    """The whole point: declaration -> resolve -> capture -> read.

    This is the case that raised ``Unknown component
    'membrane_force_xx'`` before the wiring, and it asserts the values
    are non-zero because a slab of zeros is what the upstream shells
    return and is indistinguishable from success at every other layer.
    """
    fem = _build_model("ASDShellQ4", with_brick=False)
    spec = DomainCaptureSpec()
    spec.gauss(components="shell_resultant", ids=[_SHELL_EID], name="shell")

    path = tmp_path / "cap.h5"
    _capture(spec, fem, path)

    from apeGmsh.results import Results
    with Results.from_native(
        path, fem=fem, model=_open_model_from_h5(path),
    ) as r:
        stage = r.stage(r.stages[0].id)
        peak = {}
        for name in SHELL_STRESS_RESULTANTS:
            slab = stage.elements.gauss.get(component=name)
            assert slab.values.shape == (1, 4), name
            peak[name] = float(np.abs(slab.values).max())

    assert max(peak.values()) > 0.0, (
        f"every shell resultant came back zero: {peak}. A correctly "
        f"sized slab of zeros is exactly what the upstream shells "
        f"return, so this is the assertion that distinguishes a real "
        f"capture from a dead per-Gauss-point probe."
    )
    # Transverse tip load on a cantilever strip — bending dominates.
    assert peak["bending_moment_xx"] > peak["membrane_force_xx"]


def test_upstream_shell_is_skipped_rather_than_recorded_as_zeros(
    tmp_path: Path,
) -> None:
    """``ShellMITC4`` records nothing, loudly, instead of zeros.

    Its ``materialPointers[i]->getStressResultant()`` never sees the
    committed state: the element solves correctly (the assertion on
    tip displacement below holds) and the probe still answers zeros.
    Recording that would contour ``wall`` one flat colour and read as
    "unstressed".
    """
    fem = _build_model("ShellMITC4", with_brick=False)
    assert abs(ops.nodeDisp(3, 3)) > 0.0, "the element itself must solve"
    assert max(abs(v) for v in ops.eleResponse(_SHELL_EID, "stresses")) == 0.0

    spec = DomainCaptureSpec()
    spec.gauss(components="shell_resultant", ids=[_SHELL_EID], name="shell")

    path = tmp_path / "cap.h5"
    with pytest.warns(UserWarning, match="returns zeros"):
        _capture(spec, fem, path)

    from apeGmsh.results import Results
    with Results.from_native(
        path, fem=fem, model=_open_model_from_h5(path),
    ) as r:
        stage_id = r.stages[0].id
        assert r._reader.available_components(
            stage_id, _gauss_level(),
        ) == [], "a skipped class must leave no slab behind"


# =====================================================================
# G4 — the picker offers the resultants only where a shell is drawn
# =====================================================================

def test_scoped_components_separate_shell_from_solid(tmp_path: Path) -> None:
    """The A6.6 picker law, on the two-family case it exists for.

    One file, two gauss records: a brick carrying ``stress_*`` and a
    shell carrying the resultants. A pane scoped to the shell must be
    offered the resultants and NOT ``stress_zz``; a pane scoped to the
    brick, the reverse. This is the synthetic fixture A6.6 asked for —
    the bench is too expensive to be a gate.
    """
    fem = _build_model("ASDShellQ4", with_brick=True)
    spec = DomainCaptureSpec()
    spec.gauss(components="shell_resultant", ids=[_SHELL_EID], name="shell")
    spec.gauss(components="stress", ids=[_BRICK_EID], name="solid")

    path = tmp_path / "cap.h5"
    _capture(spec, fem, path)

    from apeGmsh.results import Results
    from apeGmsh.viewers.session._realize import _components_in_scope
    with Results.from_native(
        path, fem=fem, model=_open_model_from_h5(path),
    ) as r:
        stage_id = r.stages[0].id
        # Asked through the SAME helper the inspector's pickers use,
        # not the reader directly — a reader that answers correctly
        # behind a picker that never calls it is the A6.6 defect.
        whole = set(r._reader.available_components(
            stage_id, _gauss_level(),
        ))
        on_shell = _components_in_scope(
            r, stage_id, np.array([_SHELL_EID], dtype=np.int64),
        )["gauss"]
        on_solid = _components_in_scope(
            r, stage_id, np.array([_BRICK_EID], dtype=np.int64),
        )["gauss"]

    resultants = set(SHELL_STRESS_RESULTANTS)
    assert resultants <= whole and "stress_zz" in whole, (
        f"the fixture must record both families; got {sorted(whole)}"
    )
    assert on_shell == resultants
    assert "stress_zz" not in on_shell
    assert "stress_zz" in on_solid
    assert not (resultants & on_solid)

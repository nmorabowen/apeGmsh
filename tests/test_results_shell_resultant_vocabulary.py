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
* the honesty half — after a step solved with ``algorithm Linear`` the
  upstream shells (``ShellMITC4`` &c.) answer
  ``ops.eleResponse(eid, "stresses")`` with a correctly sized vector of
  ZEROS (or, from the second step on, the previous step's state), so
  making the vocabulary reachable would have made a flat contour the
  default shell picture. DomainCapture skips those classes with a
  reason instead. Under Newton the same probes are correct (#1992),
  which the ``live`` cases at the end of this file measure.

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
from tests.fixtures.capture_tags import identity_tags  # noqa: E402


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
    with DomainCapture(resolved, path, fem,
                       tag_map=identity_tags(resolved)) as cap:
        cap.begin_stage("static_load", kind="static")
        cap.step(t=ops.getTime())
        cap.end_stage()


def _gauss_level():
    from apeGmsh.results.readers._protocol import ResultLevel
    return ResultLevel.GAUSS


@pytest.mark.live
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


@pytest.mark.live
def test_upstream_shell_is_skipped_rather_than_recorded_as_zeros(
    tmp_path: Path,
) -> None:
    """Under ``algorithm Linear``, ``ShellMITC4`` records nothing, loudly.

    ``_build_model`` solves with ``algorithm Linear``, which never
    re-forms the residual after its solve, so the sections never see
    the converged displacement: the element solves correctly (the
    assertion on tip displacement below holds) and the probe still
    answers zeros. Recording that would contour ``wall`` one flat
    colour and read as "unstressed". Under Newton the probe is correct
    and this skip discards real data; see
    ``test_domain_capture_keeps_listed_shell_under_newton``.
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

@pytest.mark.live
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


# =====================================================================
# #1992 — the stale probe is an algorithm-Linear effect, on every route
# =====================================================================
#
# One ElasticMembranePlateSection element (two triangles for the T
# classes) on the unit square, in uniaxial membrane tension with free
# lateral contraction: a constant-strain state every one of these
# elements reproduces exactly, so the closed form is N11 = 2·P·λ on a
# unit edge, with no mesh error. Each step is read on three routes that
# exist on stock openseespy: ``eleResponse "stresses"``,
# ``eleResponse "section" 1 "forces"`` and a ``.out`` Element recorder.

_P_NODE = 1.0e4                 # per loaded node; N11 = 2.0e4 at λ = 1
_N11_FULL = 2.0 * _P_NODE

# Linear reports step k-1's state at step k (zero at step 1).
_LAGGING_UNDER_LINEAR = ("ShellMITC4", "ShellDKGQ", "ShellDKGT")
# Linear reports zero at every step.
_ZERO_UNDER_LINEAR = ("ShellNLDKGQ", "ShellNLDKGT")
# The NL classes carry a geometric correction of O(strain) ~ 1e-6 at
# λ = 1; the linear-kinematics classes match the closed form to roundoff.
_RTOL = {
    "ShellMITC4": 1e-9, "ShellDKGQ": 1e-9, "ShellDKGT": 1e-9,
    "ShellNLDKGQ": 1e-5, "ShellNLDKGT": 1e-5,
}
_MEASURED = _LAGGING_UNDER_LINEAR + _ZERO_UNDER_LINEAR


def _build_uniaxial(
    shell_class: str, algorithm: str, n_steps: int,
    out_file: Path | None = None,
) -> dict[str, list[float]]:
    """Solve the uniaxial-tension square; return N11 per step per probe.

    The domain is left built (not wiped) so a caller can capture it; a
    caller reading ``out_file`` must ``ops.wipe()`` first to flush it.
    """
    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for nid, xyz in enumerate(_SHELL_COORDS, start=1):
        ops.node(nid, *(float(v) for v in xyz))
    # In-plane statically determinate: node 1 pinned, node 4 on an
    # x-roller; plate and drilling DOFs held (the exact solution has no
    # bending and zero in-plane rotation).
    ops.fix(1, 1, 1, 1, 1, 1, 1)
    ops.fix(2, 0, 0, 1, 1, 1, 1)
    ops.fix(3, 0, 0, 1, 1, 1, 1)
    ops.fix(4, 1, 0, 1, 1, 1, 1)
    ops.section("ElasticMembranePlateSection", 1, 2.0e10, 0.2, 0.2, 0.0)
    if shell_class.endswith("T"):
        ops.element(shell_class, _SHELL_EID, 1, 2, 3, 1)
        ops.element(shell_class, _SHELL_EID + 1, 1, 3, 4, 1)
    else:
        ops.element(shell_class, _SHELL_EID, 1, 2, 3, 4, 1)
    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid in (2, 3):
        ops.load(nid, _P_NODE, 0.0, 0.0, 0.0, 0.0, 0.0)
    if out_file is not None:
        ops.recorder(
            "Element", "-file", out_file.as_posix(), "-precision", 17,
            "-ele", _SHELL_EID, "stresses",
        )
    ops.system("FullGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.test("NormDispIncr", 1e-10, 10)
    ops.algorithm(algorithm)
    ops.integrator("LoadControl", 1.0 / n_steps)
    ops.analysis("Static")
    probes: dict[str, list[float]] = {"stresses": [], "section_forces": []}
    for _ in range(n_steps):
        assert ops.analyze(1) == 0
        assert ops.nodeDisp(2, 1) > 0.0, "the element itself must solve"
        probes["stresses"].append(
            float(ops.eleResponse(_SHELL_EID, "stresses")[0]))
        probes["section_forces"].append(
            float(ops.eleResponse(_SHELL_EID, "section", 1, "forces")[0]))
    return probes


def _all_routes(
    shell_class: str, algorithm: str, n_steps: int, tmp_path: Path,
) -> dict[str, np.ndarray]:
    out_file = tmp_path / f"{shell_class}_{algorithm}.out"
    probes = _build_uniaxial(shell_class, algorithm, n_steps, out_file)
    ops.wipe()                          # flushes and closes the recorder
    routes = {k: np.asarray(v) for k, v in probes.items()}
    routes[".out"] = np.loadtxt(out_file, ndmin=2)[:, 0]
    return routes


@pytest.mark.live
def test_catalog_set_is_the_measured_set() -> None:
    """The catalog note describes exactly the classes measured below."""
    from apeGmsh.opensees._response_catalog import ZERO_GAUSS_PROBE_CLASSES
    assert ZERO_GAUSS_PROBE_CLASSES == frozenset(_MEASURED)


@pytest.mark.live
@pytest.mark.parametrize("shell_class", _MEASURED)
def test_listed_shell_is_correct_under_newton(
    shell_class: str, tmp_path: Path,
) -> None:
    """Newton re-forms the residual: every route reads N11 = 2·P·λ."""
    routes = _all_routes(shell_class, "Newton", 2, tmp_path)
    expected = np.array([0.5, 1.0]) * _N11_FULL
    for route, got in routes.items():
        np.testing.assert_allclose(
            got, expected, rtol=_RTOL[shell_class], atol=0.0,
            err_msg=f"{shell_class} under Newton, route {route}",
        )


@pytest.mark.live
@pytest.mark.parametrize("shell_class", _MEASURED)
def test_listed_shell_is_stale_under_linear_on_every_route(
    shell_class: str, tmp_path: Path,
) -> None:
    """Linear never re-forms the residual: the sections hold old state.

    Step 1 is exactly zero on every route. At step 2 the linear-
    kinematics classes report step 1's state (a plausible, non-zero,
    wrong value: half the load), and the NL classes still report zero.
    The ``.out`` route agrees, so a recorder records this silently.
    """
    routes = _all_routes(shell_class, "Linear", 2, tmp_path)
    if shell_class in _LAGGING_UNDER_LINEAR:
        stale = np.array([0.0, 0.5 * _N11_FULL])
    else:
        stale = np.array([0.0, 0.0])
    for route, got in routes.items():
        assert got[0] == 0.0, f"{shell_class} Linear step 1, route {route}"
        np.testing.assert_allclose(
            got, stale, rtol=1e-9, atol=0.0,
            err_msg=f"{shell_class} under Linear, route {route}",
        )


@pytest.mark.live
@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="has_dead_gauss_probe takes only the class, so DomainCapture "
           "skips these shells under Newton too. Slice #2001 rewrites "
           "this test to pass the algorithm in.",
)
def test_domain_capture_keeps_listed_shell_under_newton(
    tmp_path: Path,
) -> None:
    """The right answer: under Newton DomainCapture records N11 = 2·P.

    Fails today because the skip is algorithm-blind. This capture has no
    bridge and no algorithm, so a fail-closed fix (unknown algorithm
    means skip) leaves it XFAIL; it does not flip by itself. Slice #2001
    rewrites it to pass the algorithm in and drops the marker.
    """
    _build_uniaxial("ShellMITC4", "Newton", 1)
    fem = _MinimalFem(np.arange(1, 5, dtype=np.int64), _SHELL_COORDS)
    spec = DomainCaptureSpec()
    spec.gauss(components="shell_resultant", ids=[_SHELL_EID], name="shell")

    path = tmp_path / "cap.h5"
    _capture(spec, fem, path)
    ops.wipe()

    from apeGmsh.results import Results
    with Results.from_native(
        path, fem=fem, model=_open_model_from_h5(path),
    ) as r:
        stage_id = r.stages[0].id
        assert "membrane_force_xx" in r._reader.available_components(
            stage_id, _gauss_level(),
        ), "DomainCapture skipped a shell whose probe is correct"
        slab = r.stage(stage_id).elements.gauss.get(
            component="membrane_force_xx",
        )
        np.testing.assert_allclose(slab.values, _N11_FULL, rtol=1e-9)

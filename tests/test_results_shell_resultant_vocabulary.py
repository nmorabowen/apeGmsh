"""The shell stress resultants are reachable from a declarative spec.

ADR 0098 A6.6 R2. ``_vocabulary.SHELL_STRESS_RESULTANTS`` named all
eight, but they were never wired into ``ALL_CANONICAL`` — so
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
  default shell picture. The bridge emit warns on those classes
  instead (#2001). Under Newton the same probes are correct (#1992),
  which the ``live`` cases at the end of this file measure.

The in-process capture route these cases once drove was deleted in
#2191; the recorder routes (``.out``, MPCO, ``.ladruno``) are measured
below and in ``tests/test_results_catalog_out_real.py``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from apeGmsh._vocabulary import (
    CATEGORY_COMPONENTS,
    SHELL_GENERALIZED_STRAINS,
    SHELL_STRESS_RESULTANTS,
    expand_shorthand,
    in_category,
    is_canonical,
)
from tests.conftest import _open_model_from_h5


class _LazyOps:
    """openseespy, imported by the first ``live`` test that touches it.

    A module-level ``importorskip`` skipped the pure vocabulary and
    predicate tests too, wherever openseespy is absent (the ``suite``
    lane). Only the ``live`` tests reach this, and they skip there.
    """

    def __getattr__(self, name: str):
        # pytest probes module globals (``__test__``, ``_pytestfixture...``)
        # at collection; an importorskip there would skip the whole module.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(pytest.importorskip("openseespy.opensees"), name)


ops = _LazyOps()


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
    assert [c for c in CATEGORY_COMPONENTS if in_category(c, name)] == [
        "gauss",
    ]


def test_gauss_shorthands_advertise_the_shell_pair() -> None:
    """The recorder declaration accepts the shell pair under ``gauss``."""
    from apeGmsh.opensees.recorder import build_recorder_declaration

    for shorthand, expanded in (
        ("shell_resultant", SHELL_STRESS_RESULTANTS),
        ("shell_deformation", SHELL_GENERALIZED_STRAINS),
    ):
        decl = build_recorder_declaration(
            ndm=3, ndf=6, pg="Wall", gauss=shorthand,
        )
        assert decl.records[0].components == expanded, shorthand


def test_resultants_and_generalized_strains_cannot_share_a_record() -> None:
    """They are work-conjugates under different ``eleResponse`` keywords."""
    from apeGmsh.results.spec._emit import _gauss_record_ops_keyword
    from apeGmsh.results.spec._resolved import ResolvedRecorderRecord

    ok = ResolvedRecorderRecord(
        category="gauss", name="r", components=SHELL_STRESS_RESULTANTS,
        dt=None, n_steps=None, element_ids=np.array([1], dtype=np.int64),
    )
    assert _gauss_record_ops_keyword(ok) == "stresses"

    mixed = ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("membrane_force_xx", "curvature_xx"),
        dt=None, n_steps=None, element_ids=np.array([1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="work-conjugate"):
        _gauss_record_ops_keyword(mixed)


# =====================================================================
# G4 — the picker offers the resultants only where a shell is drawn
# =====================================================================

_SHELL_COORDS = np.array([
    [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
], dtype=np.float64)

_SHELL_EID = 1
_BRICK_EID = 2
_STRESS = ("stress_xx", "stress_yy", "stress_zz",
           "stress_xy", "stress_yz", "stress_xz")


def _gauss_level():
    from apeGmsh.results.readers._protocol import ResultLevel
    return ResultLevel.GAUSS


def _two_family_file(tmp_path: Path) -> Path:
    """One stage, two gauss groups: a shell's resultants, a brick's stress.

    Written with :class:`NativeWriter`, the writer every native route
    shares. The values are placeholders: the picker law asks which
    components a scope offers, not what they read.
    """
    from apeGmsh.results.writers import NativeWriter

    path = tmp_path / "two_family.h5"
    with NativeWriter(path) as w:
        w.open()
        sid = w.begin_stage(
            name="static_load", kind="static", time=np.array([1.0]),
        )
        w.write_gauss_group(
            sid, "partition_0", "shell",
            class_tag=203, int_rule=0,
            element_index=np.array([_SHELL_EID], dtype=np.int64),
            natural_coords=np.zeros((4, 2)),
            components={
                c: np.ones((1, 1, 4)) for c in SHELL_STRESS_RESULTANTS
            },
        )
        w.write_gauss_group(
            sid, "partition_0", "solid",
            class_tag=56, int_rule=0,
            element_index=np.array([_BRICK_EID], dtype=np.int64),
            natural_coords=np.zeros((8, 3)),
            components={c: np.ones((1, 1, 8)) for c in _STRESS},
        )
        w.end_stage()
    return path


def test_scoped_components_separate_shell_from_solid(tmp_path: Path) -> None:
    """The A6.6 picker law, on the two-family case it exists for.

    One file, two gauss groups: a brick carrying ``stress_*`` and a
    shell carrying the resultants. A pane scoped to the shell must be
    offered the resultants and NOT ``stress_zz``; a pane scoped to the
    brick, the reverse. This is the synthetic fixture A6.6 asked for —
    the bench is too expensive to be a gate.
    """
    path = _two_family_file(tmp_path)

    from apeGmsh.results import Results
    from apeGmsh.viewers.session._realize import _components_in_scope
    with Results.from_native(path, model=_open_model_from_h5(path)) as r:
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

    A caller reading ``out_file`` must ``ops.wipe()`` first to flush it.
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


# =====================================================================
# #2001 — the bridge emit warns iff the recorded values are stale
# =====================================================================
#
# The same uniaxial square, built through the bridge and run live with a
# bridge-emitted ``.out`` Element recorder. The bridge emits two of the
# listed classes (ShellMITC4, ShellDKGQ). Under Linear the emit warns and
# the file holds the stale values PR #2000 pinned; under Newton the emit
# is silent (run this with ``-W error::UserWarning``) and the file holds
# N11 = 2·P·λ. The warning on the MPCO and .ladruno routes is the same
# emit-time check, covered per route in
# ``tests/opensees/unit/test_stale_shell_probe_emit_warning.py``.

def _bridge_uniaxial(shell_class: str, algorithm: str, out_file: Path):
    from apeGmsh.opensees import apeSees
    from tests.opensees.fixtures.fem_stub import (
        FEMStub,
        _ElementGroupView,
        _ElementsStub,
        _NodesStub,
    )

    fem = FEMStub(
        nodes=_NodesStub(
            ids=[1, 2, 3, 4],
            coords=[tuple(float(v) for v in xyz) for xyz in _SHELL_COORDS],
            node_pgs={"Pin": [1], "Roll": [4], "Edge": [2, 3]},
        ),
        elements=_ElementsStub(elem_pgs={
            "Wall": _ElementGroupView(ids=(1,), connectivity=((1, 2, 3, 4),)),
        }),
    )
    b = apeSees(fem)  # type: ignore[arg-type]
    b.model(ndm=3, ndf=6)
    sec = b.section.ElasticMembranePlateSection(E=2.0e10, nu=0.2, h=0.2)
    getattr(b.element, shell_class)(pg="Wall", section=sec)
    b.fix(pg="Pin", dofs=(1, 1, 1, 1, 1, 1))
    b.fix(pg="Edge", dofs=(0, 0, 1, 1, 1, 1))
    b.fix(pg="Roll", dofs=(1, 0, 1, 1, 1, 1))
    with b.pattern.Plain(series=b.timeSeries.Linear()) as p:
        p.load(pg="Edge", forces=(_P_NODE, 0.0, 0.0, 0.0, 0.0, 0.0))
    b.recorder.Element(
        file=out_file.as_posix(), response=("stresses",), pg="Wall",
        precision=17,
    )
    b.constraints.Plain()
    b.numberer.Plain()
    b.system.FullGeneral()
    b.test.NormDispIncr(tol=1e-10, max_iter=10)
    getattr(b.algorithm, algorithm)()
    b.integrator.LoadControl(dlam=0.5)
    b.analysis.Static()
    return b


def _run_bridge(b, n_steps: int) -> None:
    from apeGmsh.opensees.emitter.live import LiveOpsEmitter

    emitter = LiveOpsEmitter(wipe=True)
    b.build().emit(emitter)
    for _ in range(n_steps):
        assert emitter.analyze(steps=1) == 0
    emitter.ops.wipe()                  # flushes and closes the recorder


@pytest.mark.live
@pytest.mark.parametrize("shell_class", ("ShellMITC4", "ShellDKGQ"))
def test_bridge_linear_emit_warns_and_the_out_file_is_stale(
    shell_class: str, tmp_path: Path,
) -> None:
    from apeGmsh.opensees._internal.build.validators.shell_probe import (
        StaleShellProbeWarning,
    )

    ops.wipe()                          # binds the backend or skips
    out = tmp_path / "wall.out"
    b = _bridge_uniaxial(shell_class, "Linear", out)
    with pytest.warns(StaleShellProbeWarning, match=shell_class) as rec:
        _run_bridge(b, 2)
    assert len([w for w in rec
                if issubclass(w.category, StaleShellProbeWarning)]) == 1
    got = np.loadtxt(out, ndmin=2)[:, 0]
    np.testing.assert_allclose(got, [0.0, 0.5 * _N11_FULL], rtol=1e-9)


@pytest.mark.live
@pytest.mark.parametrize("shell_class", ("ShellMITC4", "ShellDKGQ"))
def test_bridge_newton_emit_is_silent_and_the_out_file_is_correct(
    shell_class: str, tmp_path: Path,
) -> None:
    import warnings

    from apeGmsh.opensees._internal.build.validators.shell_probe import (
        StaleShellProbeWarning,
    )

    ops.wipe()
    out = tmp_path / "wall.out"
    b = _bridge_uniaxial(shell_class, "Newton", out)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StaleShellProbeWarning)
        _run_bridge(b, 2)
    got = np.loadtxt(out, ndmin=2)[:, 0]
    np.testing.assert_allclose(
        got, np.array([0.5, 1.0]) * _N11_FULL, rtol=1e-9)


@pytest.mark.live
@pytest.mark.parametrize(("rung", "warns"), [("Linear", 1), ("KrylovNewton", 0)])
def test_bridge_flat_analyze_ladder_rung_counts(
    rung: str, warns: int, tmp_path: Path,
) -> None:
    """A flat ``analyze(strategy=)`` ladder never reaches the emit; a
    Linear rung of it must still warn, once, and an iterating one not."""
    import warnings

    from apeGmsh.opensees._internal.build.validators.shell_probe import (
        StaleShellProbeWarning,
    )

    ops.wipe()
    b = _bridge_uniaxial("ShellMITC4", "Newton", tmp_path / "wall.out")
    from apeGmsh.opensees.analysis import algorithm as algo_mod
    # A bare rung, as ladders take them: ``b.algorithm.X()`` would also
    # register X as the flat chain's algorithm.
    ladder = b.strategy.Ladder(rungs=[getattr(algo_mod, rung)()])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert b.analyze(steps=1, strategy=ladder) == 0
    hits = [w for w in caught
            if issubclass(w.category, StaleShellProbeWarning)]
    assert len(hits) == warns, [str(w.message) for w in hits]
    ops.wipe()

"""ADR 0091 — the consistent-load basis vs element-family guard.

Consistent reductions stamp their shape-function ``basis`` on every
:class:`NodalLoadRecord` they emit.  At ``build()``,
``validate_load_basis_vs_elements`` cross-references imported records
against the deck's element declarations and warns (fail-soft,
:class:`WarnLoadBasisMismatch`) when

* Lagrange-consistent loads land on Bézier CONTROL-value elements
  (BezierTet10 / BezierTri6 — the TIMs T2 strip-footing mechanism), or
* Bernstein-consistent loads land on nodal-value elements (the
  mirrored mismatch).

Basis-less records (point / tributary / resultants / gravity equal
split) and interface nodes shared by both families are exempt.

Loads are injected directly onto ``fem.nodes.loads`` (the resolver
output) so each test controls the basis tag exactly — mirroring
``test_body_force_double_count.py``.
"""
from __future__ import annotations

import warnings as _warnings

import pytest

from apeGmsh import apeGmsh
from apeGmsh._kernel.record_sets import NodalLoadSet
from apeGmsh._kernel.records._loads import NodalLoadRecord
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import (
    WarnLoadBasisMismatch,
    validate_load_basis_vs_elements,
)
from apeGmsh.opensees.emitter.recording import RecordingEmitter
from apeGmsh.opensees.material.nd import ElasticIsotropic


@pytest.fixture(scope="module")
def tet10_fem():
    """A small tet10 box with one volume PG — the Bézier-capable mesh."""
    g = apeGmsh(model_name="basis_mismatch", verbose=False)
    g.begin()
    try:
        g.model.geometry.add_box(0.0, 0.0, 0.0, 1.0, 1.0, 1.0, label="soil")
        g.physical.add(3, "soil", name="soil")
        g.mesh.sizing.set_global_size(0.7)
        g.mesh.generation.generate(dim=3)
        g.mesh.generation.set_order(2, bubble=False)
        yield g.mesh.queries.get_fem_data(dim=3)
    finally:
        g.end()


def _inject_loads(fem, *, basis, pattern="press"):
    recs = [
        NodalLoadRecord(
            node_id=int(n), force_xyz=(0.0, 0.0, -1.0),
            pattern=pattern, basis=basis,
        )
        for n in fem.nodes.ids
    ]
    fem.nodes.loads = NodalLoadSet(recs)


def _author(fem, *, element, from_model_case="press"):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.register(ElasticIsotropic(E=1.0e7, nu=0.25))
    getattr(ops.element, element)(pg="soil", material=mat)
    if from_model_case is not None:
        with ops.pattern.Plain(series=ops.timeSeries.Linear()) as p:
            p.from_model(from_model_case)
    return ops


def _assert_silent(ops):
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", WarnLoadBasisMismatch)
        ops.build().emit(RecordingEmitter())


# ---------------------------------------------------------------------
# The four basis × family quadrants
# ---------------------------------------------------------------------

def test_lagrange_loads_on_bezier_warn(tet10_fem):
    """The T2 mechanism: Lagrange-consistent loads on control values."""
    _inject_loads(tet10_fem, basis="lagrange")
    ops = _author(tet10_fem, element="BezierTet10")
    with pytest.warns(WarnLoadBasisMismatch, match="bernstein"):
        ops.build().emit(RecordingEmitter())


def test_bernstein_loads_on_bezier_are_silent(tet10_fem):
    _inject_loads(tet10_fem, basis="bernstein")
    _assert_silent(_author(tet10_fem, element="BezierTet10"))


def test_bernstein_loads_on_nodal_elements_warn(tet10_fem):
    """The mirrored mismatch: equal control-point loads on a Lagrange
    tet10 are the wrong local distribution too."""
    _inject_loads(tet10_fem, basis="bernstein")
    ops = _author(tet10_fem, element="TenNodeTetrahedron")
    with pytest.warns(WarnLoadBasisMismatch, match="nodal-value"):
        ops.build().emit(RecordingEmitter())


def test_lagrange_loads_on_nodal_elements_are_silent(tet10_fem):
    _inject_loads(tet10_fem, basis="lagrange")
    _assert_silent(_author(tet10_fem, element="TenNodeTetrahedron"))


# ---------------------------------------------------------------------
# Exemptions
# ---------------------------------------------------------------------

def test_basisless_records_are_exempt(tet10_fem):
    """Point / tributary / resultant / gravity records carry basis=None
    and never trip the guard — even on a Bézier deck."""
    _inject_loads(tet10_fem, basis=None)
    _assert_silent(_author(tet10_fem, element="BezierTet10"))


def test_unimported_case_is_exempt(tet10_fem):
    """Records that no pattern imports never reach the deck — silent."""
    _inject_loads(tet10_fem, basis="lagrange")
    _assert_silent(
        _author(tet10_fem, element="BezierTet10", from_model_case=None)
    )


def test_interface_nodes_shared_by_both_families_are_exempt(tet10_fem):
    """Exclusive-ownership rule: a node covered by BOTH families (an
    interface node) never counts — direct-call check with stub specs
    whose PGs both resolve to the whole box."""
    class BezierTet10:      # class NAME is what the guard dispatches on
        pg = "soil"

    class TenNodeTetrahedron:
        pg = "soil"

    _inject_loads(tet10_fem, basis="lagrange")
    with _warnings.catch_warnings():
        _warnings.simplefilter("error", WarnLoadBasisMismatch)
        validate_load_basis_vs_elements(
            tet10_fem, [BezierTet10(), TenNodeTetrahedron()], ["press"],
        )


# ---------------------------------------------------------------------
# Message quality + LadrunoUP Taylor–Hood coverage
# ---------------------------------------------------------------------

def test_warning_names_case_and_remedy(tet10_fem):
    _inject_loads(tet10_fem, basis="lagrange", pattern="surcharge")
    ops = _author(tet10_fem, element="BezierTet10",
                  from_model_case="surcharge")
    with pytest.warns(WarnLoadBasisMismatch) as rec:
        ops.build().emit(RecordingEmitter())
    msg = str(rec[0].message)
    assert "surcharge" in msg and "basis='bernstein'" in msg


def test_ladruno_up_tet10_counts_as_control_value(tet10_fem):
    """LadrunoUP on a tet10 mesh is the Bézier Taylor–Hood variant
    (quadratic Bernstein u) — Lagrange-consistent loads must warn.
    Direct call so the u-p solver/ndf gates stay out of scope."""
    class LadrunoUP:
        pg = "soil"

    _inject_loads(tet10_fem, basis="lagrange")
    with pytest.warns(WarnLoadBasisMismatch, match="bernstein"):
        validate_load_basis_vs_elements(
            tet10_fem, [LadrunoUP()], ["press"],
        )

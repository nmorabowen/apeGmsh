"""ADR 0093 S10 — the acceptance battery, deck-level half.

S10 formalises the requester's own trust checks as permanent tests. The
battery is split across two files by what it *needs*, not by what it
covers:

* **this file** — everything provable from the emitted deck and the
  resolved records, so it runs everywhere, on every CI lane, with no
  engine;
* ``tests/opensees/subprocess/test_interface_acceptance_engine.py`` —
  the checks only a running solver can answer (bonded limit, unilateral
  both signs on the curved master, slip saturation, rotated-compose
  mechanics, MPCO springs channels), gated on
  ``APEGMSH_OPENSEES_BIN``.

The acceptance list from the ADR, and where each item lives:

===========================================  ==========================
item                                         home
===========================================  ==========================
bonded limit reproduces a bilateral tie      engine file
unilateral zero-tension, both signs,         engine file
curved master, ``ent`` **and** ``epp_gap``
slip saturation Σ = τ_b·L·t                  engine file (Σ on the
                                             engine) + **here** (the
                                             per-pair closed form read
                                             off the emitted materials)
INV-3 tributary closure                      **here**, on an UNEVEN mesh
h5 round-trip → emit byte-identity           **here**
compose invariance (INV-2)                   engine file
1-vs-N rank identity                         already owned by
                                             ``tests/opensees/subprocess/
                                             test_interface_partitioned_
                                             numeric_twin.py`` (serial vs
                                             2-rank OpenSeesMP, REL_TOL
                                             1e-10) and by the byte-level
                                             ``test_interface_partitioned_
                                             emit.py``; **not duplicated**
MPCO springs channels readable per pair      engine file (live) +
                                             **here** (the recorder
                                             declaration)
===========================================  ==========================

Two of these deserve a word on *why* they are asserted from the deck
rather than from the resolver's own output.

``A_trib`` closure (INV-3) is asserted by the resolver itself at resolve
time. Re-asserting it from the resolver's records would only re-run the
resolver's own arithmetic; the battery therefore reconstructs each
pair's tributary area from the **emitted material line** (``E / k`` for
an ``ENT`` normal law), so a bug in the emit-time translation — the one
layer the resolver's assertion cannot see — cannot hide behind it. Same
reasoning for the slip cap: ``ElasticPP`` takes a yield *strain*, so the
physical per-pair yield force is the emergent product ``E × epsyP`` of
two numbers that live on the deck line, and that product is what gets
checked against ``τ_b × A_trib``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh._kernel.records._constraints import NormalLaw, TangentialLaw
from apeGmsh.mesh.FEMData import FEMData
from apeGmsh.opensees import apeSees

from tests.opensees._helpers.interface_fixtures import tunnel_geometry

K_N = 1.0e9
K_T = 1.0e8
TAU_B = 2.5e5
THICKNESS = 0.5
FACE_LENGTH = 1.0                       # the master edge of the unit square

NORMAL = NormalLaw(kind="ent", k_per_area=K_N)
TANGENTIAL = TangentialLaw(kind="epp", k_per_area=K_T, tau_b=TAU_B)


# =====================================================================
# Fixtures
# =====================================================================

def _curve_at_x(surface: int, x: float, tol: float = 1e-6) -> int:
    for dim, tag in gmsh.model.getBoundary([(2, surface)], oriented=False):
        bb = gmsh.model.getBoundingBox(1, abs(tag))
        if abs(bb[0] - x) < tol and abs(bb[3] - x) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary curve of surface {surface} at x={x}")


def _two_squares(g, n: int = 4):
    """The battery's reference shape: left square ``[0,1]^2`` (the
    continuum master) against right square ``[1,2]^2``, un-fragmented,
    so the two curves at ``x=1`` carry coincident-but-distinct nodes."""
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    g.mesh.structured.set_transfinite([(2, left), (2, right)], n=n)
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [_curve_at_x(left, 1.0)], name="face")
    g.physical.add(1, [_curve_at_x(right, 1.0)], name="wire")
    g.physical.add(1, [_curve_at_x(left, 0.0)], name="base")
    g.physical.add(1, [_curve_at_x(right, 2.0)], name="anchor")


def _uneven_strip(g, n: int = 6, coef: float = 1.4):
    """The same two squares, but with a deliberately NON-UNIFORM node
    distribution along the interface (geometric progression, ~3.8x
    between the shortest and the longest segment).

    ``add_rectangle`` winds counter-clockwise, so the left square's
    right edge runs bottom→top while the right square's left edge runs
    top→bottom: the mirrored ``1/coef`` on the slave side is what keeps
    the two node sets coincident (without it the verb's pairing fails
    loud, which is itself the check that this fixture is honest).
    """
    left = g.model.geometry.add_rectangle(0, 0, 0, 1, 1)
    right = g.model.geometry.add_rectangle(1, 0, 0, 1, 1)
    g.model.sync()
    master = _curve_at_x(left, 1.0)
    slave = _curve_at_x(right, 1.0)
    st = g.mesh.structured
    st.set_transfinite_curve(master, n, coef=coef)
    st.set_transfinite_curve(_curve_at_x(left, 0.0), n, coef=coef)
    st.set_transfinite_curve(slave, n, coef=1.0 / coef)
    st.set_transfinite_curve(_curve_at_x(right, 2.0), n, coef=1.0 / coef)
    for surf in (left, right):
        for dim, tag in gmsh.model.getBoundary([(2, surf)], oriented=False):
            bb = gmsh.model.getBoundingBox(1, abs(tag))
            if abs(bb[1] - bb[4]) < 1e-9:            # horizontal edge
                st.set_transfinite_curve(abs(tag), 3)
    st.set_transfinite_surface(left)
    st.set_transfinite_surface(right)
    st.set_recombine([(2, left), (2, right)])
    g.mesh.generation.generate(2)
    g.physical.add(2, [left], name="rock")
    g.physical.add(2, [right], name="liner")
    g.physical.add(1, [master], name="face")
    g.physical.add(1, [slave], name="wire")


def _fem(*, builder=_two_squares, slave_ndf=None, **kw):
    with apeGmsh(model_name="s10_battery", verbose=False) as g:
        builder(g, **kw)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, slave_ndf=slave_ndf, name="RockLiner")
        return g.mesh.queries.get_fem_data()


def _quad_ops(fem) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    for pg in ("rock", "liner"):
        ops.element.FourNodeQuad(
            pg=pg, thickness=THICKNESS, material=mat,
            plane_type="PlaneStrain")
    return ops


def _mixed_ops(fem) -> apeSees:
    ops = apeSees(fem)
    ops.model(ndm=2, ndf=2)
    mat = ops.nDMaterial.ElasticIsotropic(E=30e9, nu=0.2, rho=2400)
    ops.element.FourNodeQuad(
        pg="rock", thickness=THICKNESS, material=mat,
        plane_type="PlaneStrain")
    ops.element.elasticBeamColumn(
        pg="wire", transf=ops.geomTransf.Linear(),
        A=0.02, E=200e9, Iz=1.0e-4)
    return ops


def _deck_text(ops: apeSees, path) -> str:
    ops.tcl(str(path))
    return path.read_text(encoding="utf-8")


def _material_args(text: str, token: str) -> "list[list[float]]":
    """Every ``uniaxialMaterial <token> <tag> <args...>`` line's args,
    in deck order (the tag is dropped)."""
    out = []
    for ln in text.splitlines():
        tok = ln.strip().split()
        if len(tok) >= 3 and tok[0] == "uniaxialMaterial" and tok[1] == token:
            out.append([float(v) for v in tok[3:]])
    return out


# =====================================================================
# Acceptance — slip cap, per pair, off the emitted materials
# =====================================================================

def test_per_pair_slip_force_is_tau_b_times_a_trib(tmp_path) -> None:
    """The tangential closed form, pair by pair, read off the deck.

    ``ElasticPP`` takes a yield **strain** (ADR 0093 D1, the review's
    correction), so no single number on the deck line *is* the yield
    force. The physical cap is the product ``E × epsyP`` — and that
    product must equal ``τ_b × A_trib`` on every pair, with ``epsyP``
    identical across pairs because ``A_trib`` cancels out of it.

    The engine-side half of this check (Σ over the pairs saturating a
    real analysis at ``τ_b × L × t``) lives in the engine file; this one
    proves the *translation* is right even where no solver runs.
    """
    fem = _fem()
    recs = fem.elements.interfaces
    assert len(recs) == 4                       # n=4 ⇒ 4 nodes on the face
    text = _deck_text(_quad_ops(fem), tmp_path / "epp.tcl")

    args = _material_args(text, "ElasticPP")
    assert len(args) == len(recs)

    # epsyP is a pure material property here: tau_b / k_per_area.
    eps = {round(a[1], 15) for a in args}
    assert eps == {TAU_B / K_T}, f"epsyP drifted across pairs: {eps}"

    a_tribs = sorted(float(r.a_trib) for r in recs)
    caps = sorted(a[0] * a[1] for a in args)                # E x epsyP
    for cap, a_trib in zip(caps, a_tribs):
        assert cap == pytest.approx(TAU_B * a_trib, rel=1e-14), (
            f"per-pair slip force {cap!r} != tau_b*A_trib "
            f"{TAU_B * a_trib!r}")

    # …and their sum is the strip's closed form, the same number the
    # engine file measures as a reaction: 2.5e5 * 1.0 * 0.5 = 1.25e5 N.
    assert sum(caps) == pytest.approx(
        TAU_B * FACE_LENGTH * THICKNESS, rel=1e-14)


# =====================================================================
# Acceptance — INV-3 tributary closure on an UNEVEN mesh
# =====================================================================

def test_tributary_closure_survives_an_uneven_mesh(tmp_path) -> None:
    """INV-3 re-asserted from the emitted ``ENT`` lines, on a mesh whose
    interface spacing varies by ~4x.

    A uniform mesh cannot distinguish ``A_trib = ℓ_trib × t`` from
    ``A_trib = L × t / n`` — every pair carries the same number either
    way. This fixture can: the segments run ~0.091 → ~0.351, so the
    per-pair areas must differ, the two END pairs must carry a half
    share each, and the sum must still close on ``L × t``.

    Reconstructing ``A_trib = E / k_per_area`` from the deck (rather
    than reading ``rec.a_trib``) is the point — the resolver already
    asserts its own closure, so only the emit-time translation is on
    trial here.
    """
    fem = _fem(builder=_uneven_strip)
    recs = fem.elements.interfaces
    coords = {int(i): tuple(map(float, c))
              for i, c in zip(fem.nodes.ids, fem.nodes.coords)}
    ys = sorted(coords[int(r.master_node)][1] for r in recs)
    spans = [b - a for a, b in zip(ys, ys[1:])]
    assert max(spans) / min(spans) > 3.0, (
        f"fixture is not uneven enough to be a real test: {spans}")

    text = _deck_text(_quad_ops(fem), tmp_path / "uneven.tcl")
    ent = _material_args(text, "ENT")
    assert len(ent) == len(recs)

    a_from_deck = sorted(a[0] / K_N for a in ent)
    # Closure — a sum of n floats against an independently computed
    # length is O(n*eps) relative, per INV-3.
    assert sum(a_from_deck) == pytest.approx(
        FACE_LENGTH * THICKNESS, rel=len(recs) * 1e-15, abs=0.0)

    # Half shares at the two polyline ends, full shares inside.
    expect = sorted(
        [0.5 * spans[0] * THICKNESS, 0.5 * spans[-1] * THICKNESS]
        + [0.5 * (a + b) * THICKNESS for a, b in zip(spans, spans[1:])]
    )
    np.testing.assert_allclose(a_from_deck, expect, rtol=1e-13, atol=0.0)


# =====================================================================
# Acceptance — h5 round-trip → emit byte-identity
# =====================================================================

@pytest.mark.parametrize(
    "slave_ndf, ops_builder", [(None, _quad_ops), (3, _mixed_ops)],
    ids=["equal_ndf", "mixed_ndf"])
def test_h5_roundtrip_emits_a_byte_identical_deck(
        tmp_path, slave_ndf, ops_builder) -> None:
    """``build → to_h5 → from_h5 → emit`` must reproduce the direct
    emit **byte for byte**, for the equal-ndf model and for the
    mixed-ndf one (phantom node + nested ``equalDOF``, the S6 sharp
    edges: the phantom tag rides the node rewrite, ``backing_element``
    rides the element rewrite).

    Field-exact record round-tripping is already pinned in
    ``tests/mesh/test_interface_h5_roundtrip.py``. This is the
    consequence a user actually sees: a persisted model emits the same
    deck as the live one. Byte-identity is the right gate because
    anything weaker (record equality, line-set equality) would green a
    reordering — and emit order *is* semantics here: the per-pair unit
    must stay phantom → equalDOF → materials → zeroLength.
    """
    fem = _fem(slave_ndf=slave_ndf)
    direct = _deck_text(ops_builder(fem), tmp_path / "direct.tcl")

    h5 = tmp_path / "model.h5"
    fem.to_h5(str(h5))
    reloaded = FEMData.from_h5(str(h5))
    assert len(reloaded.elements.interfaces) == len(fem.elements.interfaces)
    round_tripped = _deck_text(
        ops_builder(reloaded), tmp_path / "roundtrip.tcl")

    assert round_tripped == direct


# =====================================================================
# Acceptance — the MPCO springs recorder declaration
# =====================================================================

def test_mpco_springs_recorder_declaration_emits(tmp_path) -> None:
    """The deck-level half of the results check (ADR 0093 D6).

    Covered here: that the recorder asking for the two spring channels
    lands on the deck, in a shape the interface's ``zeroLength``
    elements answer. **Not** covered here — that the file written by a
    real run actually carries per-pair ``spring_force_0`` /
    ``spring_deformation_0`` matching ``eleResponse``: that needs a
    solver and lives in the engine file
    (``test_mpco_springs_channels_match_the_engine``), which skips when
    ``APEGMSH_OPENSEES_BIN`` is unset.
    """
    fem = _fem()
    ops = _quad_ops(fem)
    ops.recorder.MPCO(
        file="springs.mpco",
        nodal_responses=("displacement",),
        elem_responses=("basicForce", "deformation"),
    )
    text = _deck_text(ops, tmp_path / "rec.tcl")

    rec = [ln.strip() for ln in text.splitlines()
           if ln.strip().startswith("recorder mpco ")]
    assert len(rec) == 1, text
    assert "basicForce" in rec[0] and "deformation" in rec[0]
    # No region filter was asked for, so the whole model — including
    # every interface zeroLength — is recorded.
    assert "-R " not in rec[0]
    assert len([ln for ln in text.splitlines()
                if ln.strip().startswith("element zeroLength ")]) == 4


# =====================================================================
# Acceptance — the record set the whole battery leans on
# =====================================================================

def test_curved_master_orientation_swings_with_the_face(tmp_path) -> None:
    """The precondition the engine file's tunnel cases stand on, pinned
    where no solver is needed: on the quarter-annulus tunnel master the
    per-pair local-x vectors sweep the full 90° of the arc and every one
    of them points INTO the opening (ADR 0093 D2 / INV-1).

    The kernel suite proves the same rule against synthetic arrays
    (``tests/_kernel/resolvers/test_interface_resolver.py``, including
    ``test_curved_master_defeats_a_single_face_average_frame``); this
    one proves it survives a real Gmsh model and reaches the deck as a
    per-element ``-orient``.
    """
    fem = _tunnel_fem()
    recs = fem.elements.interfaces
    coords = {int(i): tuple(map(float, c))
              for i, c in zip(fem.nodes.ids, fem.nodes.coords)}

    angles = []
    for r in recs:
        p = np.array(coords[int(r.master_node)][:2])
        inward = -p / np.linalg.norm(p)
        n = np.array(r.orient[:2])
        assert float(n @ inward) > 0.99, (
            f"pair at {p} has local-x {n} — not pointing into the opening")
        angles.append(math.degrees(math.atan2(p[1], p[0])))
    assert max(angles) - min(angles) == pytest.approx(90.0, abs=1e-9)

    text = _deck_text(_quad_ops(fem), tmp_path / "tunnel.tcl")
    orients = set()
    for ln in text.splitlines():
        tok = ln.strip().split()
        if tok[:2] == ["element", "zeroLength"]:
            i = tok.index("-orient")
            orients.add(tuple(tok[i + 1:i + 7]))
    assert len(orients) == len(recs), (
        "the deck collapsed the per-pair frames — a face-average "
        f"implementation would show far fewer than {len(recs)}")


def _tunnel_fem():
    """Quarter-annulus tunnel: rock ``r∈[1,2]`` around the opening, a
    liner ``r∈[0.85,1]`` inside it, coincident node sets on ``r=1``.

    The master is the rock's INNER rim — a *concave* master with the
    material outside it, so outward-of-material points radially inward
    (the Cerro Lindo shape, and the curvature mirror of a convex arc).
    """
    with apeGmsh(model_name="s10_tunnel", verbose=False) as g:
        tunnel_geometry(g)
        g.constraints.interface(
            "face", "wire", normal=NORMAL, tangential=TANGENTIAL,
            thickness=THICKNESS, name="RockLiner")
        return g.mesh.queries.get_fem_data()

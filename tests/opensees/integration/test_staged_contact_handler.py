"""ADR 0092 — a staged contact model must keep an EFFECTIVE contact handler.

The staged **partitioned** refusal (S4) names its own reason: "the
partitioned staged pipeline skips the analysis-chain auto-emit (each
stage declares its own), so the forced ``LadrunoContact`` handler would
never be emitted and the interaction would be silently unenforced."

That reasoning is not partition-specific. The SERIAL staged path had the
same hole, and it hid better: the global auto-emit *does* emit
``constraints LadrunoContact``, so the deck LOOKS covered. It is not —
a stage re-declares the entire analysis chain, so the stage's own
``constraints`` line lands AFTER the global one and BEFORE the stage's
``analysis``, and OpenSees builds the analysis with whatever is current
at that moment. Measured, pre-fix, on the deck this file builds::

    contact 1 1 2 …             ← the interaction
    constraints LadrunoContact  ← global auto-emit
    constraints Transformation  ← the stage's handler
    analysis Static             ← constructed with Transformation

The contact FE adapters are never injected: the interaction does
nothing, and nothing says so. ``s.analysis()`` *requires*
``constraints=``, so the wrong handler is always reachable — there is no
"just don't declare one" escape.

The guard is :meth:`BuiltModel._validate_staged_contact_handlers`, the
contact twin of the ADR 0068 Open-item-5 EQ guard.

Emit-time only — no fork build needed.
"""
from __future__ import annotations

import warnings

import pytest

import gmsh
from apeGmsh import apeGmsh
from apeGmsh.opensees import apeSees
from apeGmsh.opensees._internal.build import BridgeError


def _face_at_z(volume_tag: int, z: float, tol: float = 1e-4) -> int:
    for dim, tag in gmsh.model.getBoundary([(3, volume_tag)], oriented=False):
        if dim != 2:
            continue
        com = gmsh.model.occ.getCenterOfMass(2, abs(tag))
        if abs(com[2] - z) < tol:
            return abs(tag)
    raise AssertionError(f"no boundary face of vol {volume_tag} at z={z}")


@pytest.fixture(scope="module")
def contact_fem():
    """Two stacked single-hex blocks with one contact interaction."""
    with apeGmsh(model_name="staged_contact_guard", verbose=False) as g:
        z0 = 1.0 - 1e-3
        b1 = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        b2 = g.model.geometry.add_box(0, 0, z0, 1, 1, 1)
        g.model.sync()
        master, slave = _face_at_z(b1, 1.0), _face_at_z(b2, z0)
        base, top = _face_at_z(b1, 0.0), _face_at_z(b2, z0 + 1.0)
        g.mesh.structured.set_transfinite_box(b1, n=2)
        g.mesh.structured.set_transfinite_box(b2, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [b1, b2], name="solid")
        g.physical.add(2, [master], name="master")
        g.physical.add(2, [slave], name="slave")
        g.physical.add(2, [base], name="base")
        g.physical.add(2, [top], name="top")
        g.constraints.contact(
            "master", "slave", formulation="nts", kn=1.0e9,
            outward=(0.0, 0.0, 1.0), name="stg",
        )
        return g.mesh.queries.get_fem_data(dim=3)


def _staged_ops(fem, handler_factory):
    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e7, nu=0.0)
    ops.element.stdBrick(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    for pg in ("master", "slave", "top"):
        ops.fix(pg=pg, dofs=(1, 1, 0))
    with ops.stage(name="load") as s:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=25),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=handler_factory(ops),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    return ops


@pytest.mark.parametrize(
    "handler_name",
    ["Transformation", "Plain", "Penalty", "Lagrange"],
)
def test_staged_contact_with_a_non_contact_handler_refuses_named(
    contact_fem, tmp_path, handler_name,
):
    """Every non-contact handler a stage could declare is refused."""
    def factory(ops):
        ctor = getattr(ops.constraints, handler_name)
        if handler_name == "Penalty":
            return ctor(alpha_sp=1e12, alpha_mp=1e12)
        return ctor()

    ops = _staged_ops(contact_fem, factory)
    with pytest.raises(BridgeError) as excinfo:
        ops.tcl(str(tmp_path / "deck.tcl"), flat=True)
    msg = str(excinfo.value)
    assert "stage 'load'" in msg                  # which stage
    assert handler_name in msg                    # what it declared
    assert "LadrunoContact" in msg                # the required handler
    assert "SILENTLY UNENFORCED" in msg           # what would have happened


def test_staged_contact_with_the_contact_handler_emits(contact_fem, tmp_path):
    """The correct declaration passes, and the stage's chain really does
    carry the contact handler through to its ``analysis`` line."""
    deck = tmp_path / "ok.tcl"
    ops = _staged_ops(contact_fem, lambda o: o.constraints.LadrunoContact())
    ops.tcl(str(deck), flat=True)
    lines = [ln.strip() for ln in deck.read_text(encoding="utf-8").splitlines()]

    analysis_at = lines.index("analysis Static")
    handlers_before = [
        (i, ln) for i, ln in enumerate(lines[:analysis_at])
        if ln.startswith("constraints ")
    ]
    assert handlers_before, "no constraint handler precedes 'analysis Static'"
    # The LAST handler before the analysis is what the analysis is built
    # with — that is the whole point of this guard.
    assert handlers_before[-1][1] == "constraints LadrunoContact", (
        "the effective handler at 'analysis Static' is "
        f"{handlers_before[-1][1]!r}, not 'constraints LadrunoContact' — "
        "the contact FE adapters would not be injected"
    )
    # And the interaction itself is in the deck, ahead of the analysis.
    assert any(ln.startswith("contact ") for ln in lines[:analysis_at])


def test_declaring_the_contact_handler_yourself_does_not_warn(
    contact_fem, tmp_path,
):
    """Agreement is not conflict.

    The auto-emit used to warn "a constraint handler was declared …
    overriding your handler" for ANY declaration — including
    ``LadrunoContact``, the one correct choice, which the staged guard
    now *requires* on every stage. Crying wolf there would put a warning
    on every correct staged contact model.
    """
    ops = _staged_ops(contact_fem, lambda o: o.constraints.LadrunoContact())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ops.tcl(str(tmp_path / "quiet.tcl"), flat=True)
    handler_warnings = [
        str(w.message) for w in caught
        if "constraint handler" in str(w.message)
    ]
    assert not handler_warnings, (
        f"declaring LadrunoContact warned: {handler_warnings}")


def test_non_contact_staged_model_is_unaffected(tmp_path):
    """The guard is contact-scoped — a staged model with no interaction
    keeps declaring whatever handler it likes."""
    with apeGmsh(model_name="staged_no_contact", verbose=False) as g:
        box = g.model.geometry.add_box(0, 0, 0, 1, 1, 1)
        g.model.sync()
        g.mesh.structured.set_transfinite_box(box, n=2)
        g.mesh.generation.generate(3)
        g.physical.add(3, [box], name="solid")
        g.physical.add(2, [_face_at_z(box, 0.0)], name="base")
        fem = g.mesh.queries.get_fem_data(dim=3)

    ops = apeSees(fem)
    ops.model(ndm=3, ndf=3)
    mat = ops.nDMaterial.ElasticIsotropic(E=2.0e7, nu=0.0)
    ops.element.stdBrick(pg="solid", material=mat)
    ops.fix(pg="base", dofs=(1, 1, 1))
    with ops.stage(name="load") as s:
        s.analysis(
            test=ops.test.NormDispIncr(tol=1e-8, max_iter=25),
            algorithm=ops.algorithm.Newton(),
            integrator=ops.integrator.LoadControl(dlam=1.0),
            constraints=ops.constraints.Transformation(),
            numberer=ops.numberer.RCM(),
            system=ops.system.UmfPack(),
            analysis=ops.analysis.Static(),
        )
        s.run(n_increments=1)
    deck = tmp_path / "no_contact.tcl"
    ops.tcl(str(deck), flat=True)          # must not raise
    assert "constraints Transformation" in deck.read_text(encoding="utf-8")

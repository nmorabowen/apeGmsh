"""Unit battery for the SOFT-family knob predicate (ADR 0092 S3, INV-3).

Pure kernel function: given a resolved contact record, name the active
SOFT-family knobs (``soft`` / ``edge_soft``) whose emitted tokens
(``-soft`` / ``-edgeSoft``) the partitioned emit path must refuse —
k_soft = SOFSCL*4*m_eff/dt^2 needs the assembled mass of BOTH surfaces
and the ghosted side contributes none on the owner rank (fork ADR-78 D4).
"Active" mirrors ``contact_args`` exactly: any value other than
``None``/``False`` emits the token — including the builder's
``edge_edge`` gate (edge knobs are dropped wholesale when ``edge_edge``
is falsy; 2026-08-13 review F5). No Gmsh, no OpenSees, no emit.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.records._constraints import ContactPlaneRecord, ContactRecord
from apeGmsh._kernel.resolvers._contact_ownership import soft_family_knobs


def _contact(**kw) -> ContactRecord:
    return ContactRecord(
        kind="contact",
        formulation=kw.pop("formulation", "nts"),
        master_faces=np.asarray([[1, 2, 3, 4]], dtype=np.int64),
        master_nps=4,
        slave_nodes=[5, 6],
        **kw,
    )


def _plane(**kw) -> ContactPlaneRecord:
    return ContactPlaneRecord(
        kind="contact_plane",
        slave_nodes=[1, 2, 3],
        normal=(0.0, 0.0, 1.0),
        point=(0.0, 0.0, 0.0),
        kn=1.0e9,
        **kw,
    )


class TestContactRecord:
    def test_no_soft_knobs_is_empty(self) -> None:
        assert soft_family_knobs(_contact(kn=1.0e6)) == ()

    def test_soft_true_bare_flag(self) -> None:
        # True = bare -soft (fork default SOFSCL 0.10) — still active.
        assert soft_family_knobs(_contact(kn="auto", soft=True)) == ("soft",)

    def test_soft_numeric_sofscl(self) -> None:
        assert soft_family_knobs(_contact(kn=1.0e6, soft=0.25)) == ("soft",)

    def test_soft_false_and_none_are_inactive(self) -> None:
        # contact_args emits nothing for False/None — neither must refuse.
        assert soft_family_knobs(_contact(kn=1.0e6, soft=False)) == ()
        assert soft_family_knobs(_contact(kn=1.0e6, soft=None)) == ()

    def test_edge_soft_alone(self) -> None:
        rec = _contact(formulation="mortar", eps_n="auto",
                       edge_edge=True, edge_soft=True)
        assert soft_family_knobs(rec) == ("edge_soft",)

    def test_edge_soft_without_edge_edge_is_dormant(self) -> None:
        # Review F5: contact_args drops ALL edge knobs when edge_edge is
        # False — no `-edgeSoft` token is ever emitted, so a dormant
        # edge_soft must not count as active (it was refused under
        # partitioning despite the serial deck carrying no token).
        rec = _contact(formulation="mortar", eps_n="auto",
                       edge_edge=False, edge_soft=0.1)
        assert soft_family_knobs(rec) == ()

    def test_soft_still_counts_when_edge_soft_is_dormant(self) -> None:
        # The edge_edge gate silences edge_soft ONLY — a live `soft=`
        # on the same record still refuses.
        rec = _contact(formulation="mortar", eps_n="auto",
                       soft=0.1, edge_edge=False, edge_soft=0.1)
        assert soft_family_knobs(rec) == ("soft",)

    def test_soft_plus_edge_soft_names_both(self) -> None:
        rec = _contact(formulation="mortar", eps_n="auto",
                       soft=0.1, edge_edge=True, edge_soft=0.2)
        assert soft_family_knobs(rec) == ("soft", "edge_soft")

    def test_visc_is_not_soft_family(self) -> None:
        # Viscous stabilisation is a damper on the owner-local active set —
        # no both-sides assembled-mass dependency, so it is NOT refused.
        assert soft_family_knobs(_contact(kn=1.0e6, visc=2.5)) == ()

    def test_kn_auto_alone_is_not_soft(self) -> None:
        # INV-3's preserved half: kn="auto" resolves natively on the owner
        # rank (master-side ownership + uncut master) and is never refused.
        assert soft_family_knobs(_contact(kn="auto")) == ()


class TestContactPlaneRecord:
    def test_no_soft_is_empty(self) -> None:
        assert soft_family_knobs(_plane()) == ()

    def test_soft_true_and_numeric(self) -> None:
        assert soft_family_knobs(_plane(soft=True)) == ("soft",)
        assert soft_family_knobs(_plane(soft=0.1)) == ("soft",)

    def test_visc_is_not_soft_family(self) -> None:
        assert soft_family_knobs(_plane(visc=1.0)) == ()


class TestErrorCases:
    def test_unsupported_record_type_raises(self) -> None:
        with pytest.raises(TypeError, match="ContactRecord or ContactPlaneRecord"):
            soft_family_knobs(object())  # type: ignore[arg-type]

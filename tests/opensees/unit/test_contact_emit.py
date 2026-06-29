"""Emit + validation contract for `g.constraints.contact` → fork contact.

Text-only (no OpenSees backend): locks the two grammar builders
(`contact_surface_args` / `contact_args`), the def validation (`ContactDef`),
and the record→emit path (`emit_contacts` → contact_surface/contact emitter
calls). Covers NTS penalty + mortar frictionless/friction + explicit `-outward`
plus the extension modifiers `-soft`/`-visc`/`-consistanttan`/`-geomtan`
(ADR 0073). The wrapper never auto-derives `-outward` — the fork derives a
correct per-facet normal otherwise (see ConstraintsComposite.resolve_contacts).
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._kernel.defs.constraints import ContactDef
from apeGmsh._kernel.records._constraints import ContactRecord
from apeGmsh.opensees._internal.build import emit_contacts
from apeGmsh.opensees._internal.tag_allocator import TagAllocator
from apeGmsh.opensees.element.contact import contact_args, contact_surface_args
from apeGmsh.opensees.emitter.recording import RecordingEmitter


# --------------------------------------------------------------------------
# contact_surface_args
# --------------------------------------------------------------------------
def test_master_faceted_with_stride():
    a = contact_surface_args("master", [1, 2, 3, 4, 5, 6], 3)
    assert a == ["-master", 3, 1, 2, 3, 4, 5, 6]


def test_slave_node_set_no_stride():
    assert contact_surface_args("slave", [7, 8]) == ["-slave", 7, 8]


def test_slave_segments_faceted():
    assert contact_surface_args("slave-segments", [3, 4, 5], 3) == \
        ["-slave-segments", 3, 3, 4, 5]


def test_faceted_rejects_bad_stride():
    with pytest.raises(ValueError, match="nps"):
        contact_surface_args("master", [1, 2, 3], 2)
    with pytest.raises(ValueError, match="multiple"):
        contact_surface_args("master", [1, 2, 3, 4], 3)


def test_faceted_rejects_higher_order_stride():
    # The fork contact subsystem only handles 3-node (tri) / 4-node (quad)
    # facets; a tri6 (nps=6) / quad8 (nps=8) stride must be rejected — higher-
    # order surfaces must be dropped to corner facets before emit.
    with pytest.raises(ValueError, match="3 .tri. or 4 .quad."):
        contact_surface_args("master", list(range(1, 13)), 6)
    with pytest.raises(ValueError, match="3 .tri. or 4 .quad."):
        contact_surface_args("slave-segments", list(range(1, 17)), 8)


# --------------------------------------------------------------------------
# contact_args
# --------------------------------------------------------------------------
def test_nts_numeric_kn_always_pads_full_triple():
    # A numeric kn always emits the full kn kt mu triple (kt/mu default 0.0 ⇒
    # frictionless). The fork's m=(remaining>=3)?3:1 reader makes a bare numeric
    # kn fragile to ANY trailing token, so we always pad — semantically
    # identical and immune to which trailing options follow.
    a = contact_args(1, 2, "nts", kn=1.0e6)
    assert a == [1, 2, 1.0e6, 0.0, 0.0]


def test_nts_friction_emits_three_numbers():
    a = contact_args(1, 2, "nts", kn=1.0e6, kt=5.0e5, mu=0.3)
    assert a == [1, 2, 1.0e6, 5.0e5, 0.3]


def test_nts_auto_kn():
    a = contact_args(1, 2, "nts", kn="auto")
    assert a == [1, 2, "auto"]


def test_nts_outward():
    a = contact_args(1, 2, "nts", kn=1.0e6, outward=(0.0, 0.0, 1.0))
    assert a[-4:] == ["-outward", 0.0, 0.0, 1.0]


def test_nts_bare_kn_plus_outward_pads_full_triple():
    # Regression: the fork's numeric kn-slot reader sizes its double read as
    # (remaining >= 3) ? 3 : 1 counting ALL trailing tokens — a bare numeric
    # kn directly followed by -outward makes it read -outward as a double and
    # abort. So a numeric kn + outward must emit the full kn kt mu triple.
    a = contact_args(1, 2, "nts", kn=1.0e6, outward=(0.0, 0.0, 1.0))
    assert a == [1, 2, 1.0e6, 0.0, 0.0, "-outward", 0.0, 0.0, 1.0]


def test_nts_auto_kn_plus_outward_not_padded():
    # The 'auto' path peeks-and-unreads the flag safely, so no padding needed.
    a = contact_args(1, 2, "nts", kn="auto", outward=(0.0, 0.0, 1.0))
    assert a == [1, 2, "auto", "-outward", 0.0, 0.0, 1.0]


def test_nts_no_kn_plus_outward_not_padded():
    # No kn emitted → the parser sees -outward as the first token and peeks-
    # and-unreads it; no leading number to pad.
    a = contact_args(1, 2, "nts", outward=(0.0, 0.0, 1.0))
    assert a == [1, 2, "-outward", 0.0, 0.0, 1.0]


def test_mortar_frictionless():
    a = contact_args(1, 2, "mortar", eps_n="auto", aug_tol=1e-8, max_aug=20, ngp=2)
    assert a[:4] == [1, 2, "-mortar", "-epsN"]
    assert "auto" in a and "-augTol" in a and "-maxAug" in a and "-ngp" in a


def test_mortar_friction_and_tie_flags():
    a = contact_args(1, 2, "mortar", eps_n=1e7, mu=0.4, cohesion=1e3, tau_max=5e5)
    assert "-mu" in a and "-cohesion" in a and "-tauMax" in a
    t = contact_args(1, 2, "mortar", eps_n=1e7, tie=True)
    assert "-tie" in t


# --------------------------------------------------------------------------
# contact_args — extension modifiers (-soft/-visc/-consistanttan/-geomtan)
# --------------------------------------------------------------------------
def test_nts_soft_bare_flag():
    # soft=True → a bare `-soft` (the fork default SOFSCL 0.10).
    a = contact_args(1, 2, "nts", kn="auto", soft=True)
    assert a == [1, 2, "auto", "-soft"]


def test_nts_soft_numeric_sofscl():
    a = contact_args(1, 2, "nts", kn="auto", soft=0.1)
    assert a == [1, 2, "auto", "-soft", 0.1]


def test_soft_false_and_none_emit_nothing():
    assert contact_args(1, 2, "nts", kn="auto", soft=False) == [1, 2, "auto"]
    assert contact_args(1, 2, "nts", kn="auto", soft=None) == [1, 2, "auto"]


def test_visc_emits_coefficient():
    a = contact_args(1, 2, "nts", kn="auto", visc=2.5)
    assert a == [1, 2, "auto", "-visc", 2.5]


def test_consistent_and_geom_tan_flags():
    a = contact_args(1, 2, "nts", kn="auto", consistent_tan=True, geom_tan=True)
    assert a == [1, 2, "auto", "-consistanttan", "-geomtan"]


def test_cell_emits_scale_both_lanes():
    # -cell <frac> is broad-phase tuning — emitted in either formulation.
    a = contact_args(1, 2, "nts", kn="auto", cell=2.0)
    assert a == [1, 2, "auto", "-cell", 2.0]
    m = contact_args(1, 2, "mortar", eps_n="auto", cell=0.5)
    assert "-cell" in m and 0.5 in m and m.index("-mortar") < m.index("-cell")


def test_cell_none_emits_nothing():
    assert contact_args(1, 2, "nts", kn="auto", cell=None) == [1, 2, "auto"]


def test_extensions_emit_before_outward():
    # The bare `-soft` must precede `-outward`; the fork peeks-and-unreads the
    # token after `-soft`, so a following flag is safe.
    a = contact_args(1, 2, "nts", kn="auto", soft=True,
                     outward=(0.0, 0.0, 1.0))
    assert a == [1, 2, "auto", "-soft", "-outward", 0.0, 0.0, 1.0]


def test_nts_all_extensions_combined():
    a = contact_args(1, 2, "nts", kn=1.0e6, kt=5.0e5, mu=0.3,
                     soft=0.2, visc=1.0, consistent_tan=True, geom_tan=True)
    assert a == [1, 2, 1.0e6, 5.0e5, 0.3,
                 "-soft", 0.2, "-visc", 1.0, "-consistanttan", "-geomtan"]


def test_mortar_soft_and_visc():
    a = contact_args(1, 2, "mortar", eps_n="auto", soft=0.1, visc=0.5)
    assert "-soft" in a and 0.1 in a and "-visc" in a and 0.5 in a
    # extensions come after the mortar block (-mortar/-epsN …)
    assert a.index("-mortar") < a.index("-soft")


# --------------------------------------------------------------------------
# contact_args — edge-edge fallback (-edgeedge + -edge*)
# --------------------------------------------------------------------------
def test_edge_edge_bare_flag():
    a = contact_args(1, 2, "mortar", eps_n="auto", edge_edge=True)
    assert "-edgeedge" in a and a.index("-mortar") < a.index("-edgeedge")


def test_edge_knobs_not_emitted_without_edge_edge():
    # the emitter drops the edge knobs when edge_edge is False (the fork ignores
    # -edge* without -edgeedge); the def is what fails loud.
    a = contact_args(1, 2, "mortar", eps_n="auto", edge_mu=0.4, edge_kn=1e7)
    assert "-edgeMu" not in a and "-edgeKn" not in a and "-edgeedge" not in a


def test_edge_kn_auto_and_numeric():
    a = contact_args(1, 2, "mortar", eps_n="auto", edge_edge=True, edge_kn="auto")
    assert a[a.index("-edgeKn") + 1] == "auto"
    n = contact_args(1, 2, "mortar", eps_n="auto", edge_edge=True, edge_kn=2e7)
    assert n[n.index("-edgeKn") + 1] == 2e7


def test_edge_soft_bare_and_numeric():
    # bare -edgeSoft (edge_soft=True) is a lone flag — the next token (if any)
    # is a flag, never a number (the fork takes its default SOFSCL 0.10).
    bare = contact_args(1, 2, "mortar", eps_n="auto", edge_edge=True,
                        edge_soft=True)
    i = bare.index("-edgeSoft")
    assert i == len(bare) - 1 or isinstance(bare[i + 1], str)
    # a numeric SOFSCL emits -edgeSoft <value>
    num = contact_args(1, 2, "mortar", eps_n="auto", edge_edge=True,
                       edge_soft=0.1)
    assert num[num.index("-edgeSoft") + 1] == 0.1


def test_edge_all_knobs_combined_in_order():
    a = contact_args(
        1, 2, "mortar", eps_n="auto", edge_edge=True,
        edge_kn=2e7, edge_band=0.01, edge_mu=0.4, edge_kt=1e6,
        edge_cohesion=1e3, edge_tau_max=5e5, edge_consistent_tan=True,
        edge_soft=0.1, edge_alm=True, edge_aug_tol=1e-6)
    for tok in ("-edgeedge", "-edgeKn", "-edgeBand", "-edgeMu", "-edgeKt",
                "-edgeCohesion", "-edgeTauMax", "-edgeConsistentTan",
                "-edgeSoft", "-edgeAlm", "-edgeAugTol"):
        assert tok in a, tok
    # -edgeedge leads the edge block
    assert a.index("-edgeedge") < a.index("-edgeKn")


def test_edge_block_precedes_outward():
    a = contact_args(1, 2, "mortar", eps_n="auto", edge_edge=True,
                     edge_mu=0.3, outward=(0.0, 0.0, 1.0))
    assert a.index("-edgeedge") < a.index("-outward")
    assert a[-4:] == ["-outward", 0.0, 0.0, 1.0]


@pytest.mark.parametrize("kw, expect_tail", [
    (dict(soft=0.1), ["-soft", 0.1]),
    (dict(soft=True), ["-soft"]),
    (dict(visc=2.5), ["-visc", 2.5]),
    (dict(consistent_tan=True), ["-consistanttan"]),
    (dict(geom_tan=True), ["-geomtan"]),
    (dict(cell=2.0), ["-cell", 2.0]),
])
def test_nts_numeric_kn_plus_extension_pads_triple(kw, expect_tail):
    # Regression (review #2): a numeric kn followed by an extension flag must
    # emit the full kn kt mu triple first — else the fork's
    # m=(remaining>=3)?3:1 reader consumes the flag as a double and aborts the
    # `contact` command. The 'auto' path peeks-and-unreads safely (not padded).
    a = contact_args(1, 2, "nts", kn=1.0e6, **kw)
    assert a == [1, 2, 1.0e6, 0.0, 0.0, *expect_tail]


# --------------------------------------------------------------------------
# ContactDef validation
# --------------------------------------------------------------------------
def test_def_formulation_validated():
    with pytest.raises(ValueError, match="formulation"):
        ContactDef(master_label="m", slave_label="s", formulation="penalty")


def test_def_nts_rejects_mortar_params():
    with pytest.raises(ValueError, match="mortar-only"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", eps_n=1e7)


def test_def_mortar_rejects_nts_params():
    with pytest.raises(ValueError, match="NTS-only"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", kn=1e6)


def test_def_tie_requires_mortar():
    with pytest.raises(ValueError, match="mortar"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", tie=True)


@pytest.mark.parametrize("formulation, extra", [
    ("nts", {}), ("mortar", {"eps_n": "auto"}),
])
def test_def_cell_must_be_positive(formulation, extra):
    # The fork wants a strictly positive broad-phase fraction; <= 0 is rejected
    # on both lanes (cell is formulation-agnostic).
    with pytest.raises(ValueError, match="cell"):
        ContactDef(master_label="m", slave_label="s",
                   formulation=formulation, cell=0.0, **extra)
    # a positive value is accepted on either lane
    d = ContactDef(master_label="m", slave_label="s",
                   formulation=formulation, cell=1.5, **extra)
    assert d.cell == 1.5


def test_def_tie_excludes_friction():
    with pytest.raises(ValueError, match="exclusive"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", tie=True, mu=0.3)


def test_def_outward_must_be_3vec():
    with pytest.raises(ValueError, match="3-vector"):
        ContactDef(master_label="m", slave_label="s", outward=(0.0, 1.0))


def test_def_outward_rejects_zero_vector():
    with pytest.raises(ValueError, match="non-zero"):
        ContactDef(master_label="m", slave_label="s",
                   outward=(0.0, 0.0, 0.0))


@pytest.mark.parametrize("kw, match", [
    (dict(formulation="nts", kn=-1.0), "kn"),
    (dict(formulation="nts", kn=1e6, kt=-1.0), "kt"),
    (dict(formulation="nts", kn=1e6, mu=-0.1), "mu"),
    (dict(formulation="mortar", eps_n=-1.0), "eps_n"),
    (dict(formulation="mortar", eps_n=1e7, eps_t=-1.0), "eps_t"),
    (dict(formulation="mortar", eps_n=1e7, cohesion=-1.0), "cohesion"),
    (dict(formulation="mortar", eps_n=1e7, tau_max=-1.0), "tau_max"),
    (dict(formulation="mortar", eps_n=1e7, aug_tol=0.0), "aug_tol"),
    (dict(formulation="mortar", eps_n=1e7, max_aug=0), "max_aug"),
    (dict(formulation="mortar", eps_n=1e7, ngp=0), "ngp"),
])
def test_def_range_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        ContactDef(master_label="m", slave_label="s", **kw)


@pytest.mark.parametrize("kw", [
    dict(formulation="nts", kn=0.0),                       # P1b zero-force path
    dict(formulation="mortar", eps_n=0.0),                 # inert mortar
    dict(formulation="mortar", eps_n=1e7, tau_max=0.0),    # "no Tresca cap"
])
def test_def_accepts_fork_zero_sentinels(kw):
    # kn==0 / eps_n==0 (the fork's documented inert / zero-force-topology path)
    # and tau_max==0 (fork "no Tresca cap" sentinel) are valid fork inputs.
    ContactDef(master_label="m", slave_label="s", **kw)


@pytest.mark.parametrize("kw, match", [
    (dict(formulation="nts", kn=float("nan")), "finite"),
    (dict(formulation="nts", kn=float("inf")), "finite"),
    (dict(formulation="mortar", eps_n=float("inf")), "finite"),
    (dict(formulation="mortar", eps_n=1e7, mu=float("nan")), "finite"),
])
def test_def_rejects_non_finite_penalty(kw, match):
    # NaN / +-inf slip past every < / == comparison and poison the tangent.
    with pytest.raises(ValueError, match=match):
        ContactDef(master_label="m", slave_label="s", **kw)


@pytest.mark.parametrize("kw, match", [
    (dict(formulation="mortar", eps_n=1e7, max_aug=2.5), "max_aug"),
    (dict(formulation="mortar", eps_n=1e7, ngp=2.5), "ngp"),
])
def test_def_rejects_non_integer_counts(kw, match):
    # a fractional Gauss order / augmentation count would be silently int()-
    # truncated at emit — fail loud instead.
    with pytest.raises(ValueError, match=match):
        ContactDef(master_label="m", slave_label="s", **kw)


def test_def_mortar_tie_requires_explicit_outward():
    # A tie interface is coincident-flat → fork gate H2 silently drops it to
    # zero force unless outward is pinned. So tie=True needs explicit outward.
    with pytest.raises(ValueError, match="tie.*outward|outward"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", tie=True, eps_n=1e7)
    # with an explicit outward it is accepted
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", tie=True, eps_n=1e7,
               outward=(0.0, 0.0, 1.0))


def test_def_auto_kn_skips_range_check():
    # "auto" is a valid sentinel and must not trip the numeric kn>0 check.
    ContactDef(master_label="m", slave_label="s",
               formulation="nts", kn="auto")
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", eps_n="auto")


def test_def_range_check_rejects_numpy_and_bool_penalty():
    # numpy scalars (routine when kn is computed via E*A/L on arrays) must be
    # range-checked, and bool must not sneak through as int.
    with pytest.raises(ValueError, match="kn"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn=np.float32(-5.0))
    with pytest.raises(ValueError, match="kn"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn=True)
    # a valid numpy float64 penalty is accepted
    ContactDef(master_label="m", slave_label="s",
               formulation="nts", kn=np.float64(1.0e6))


def test_def_rejects_bad_auto_string():
    # only the exact lowercase "auto" sentinel is valid; a typo fails at
    # construction, not opaquely in the emitter's float() later.
    with pytest.raises(ValueError, match="number or 'auto'"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn="AUTO")
    with pytest.raises(ValueError, match="number or 'auto'"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n="garbage")


# --------------------------------------------------------------------------
# ContactDef validation — extension modifiers (ADR 0073)
# --------------------------------------------------------------------------
def test_def_geomtan_is_nts_only():
    # the fork refuses -geomtan on the mortar lane.
    with pytest.raises(ValueError, match="geom_tan.*NTS-only|NTS-only"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n=1e7, geom_tan=True)
    # NTS accepts it
    ContactDef(master_label="m", slave_label="s",
               formulation="nts", kn=1e6, geom_tan=True)


def test_def_soft_excludes_tie():
    # -soft is refused with -tie (a permanent bond is not a soft penalty).
    with pytest.raises(ValueError, match="soft.*exclusive.*tie|tie"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n=1e7, tie=True,
                   outward=(0.0, 0.0, 1.0), soft=True)


@pytest.mark.parametrize("kw", [
    dict(formulation="nts", soft=True),                 # NTS, no kn
    dict(formulation="nts", kn=0.0, soft=True),         # NTS, kn=0 (inert)
    dict(formulation="mortar", soft=True),              # mortar, no eps_n
    dict(formulation="mortar", eps_n=0.0, soft=True),   # mortar, eps_n=0
])
def test_def_soft_needs_base_penalty(kw):
    # SOFT sizes k_soft under explicit but an implicit run uses the base
    # penalty — the fork aborts the contact command without one.
    with pytest.raises(ValueError, match="base penalty"):
        ContactDef(master_label="m", slave_label="s", **kw)


@pytest.mark.parametrize("kw", [
    dict(formulation="nts", kn="auto", soft=True),
    dict(formulation="nts", kn=1e6, soft=0.1),
    dict(formulation="mortar", eps_n="auto", soft=True),
    dict(formulation="mortar", eps_n=1e7, soft=0.1),
])
def test_def_soft_accepts_base_penalty(kw):
    ContactDef(master_label="m", slave_label="s", **kw)


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_def_soft_numeric_must_be_finite_positive(bad):
    with pytest.raises(ValueError, match="SOFSCL|finite"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn="auto", soft=bad)


def test_def_soft_sofscl_coupled_stability_warns():
    import warnings as _w
    # mortar SOFT=2 warns above 0.25
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n="auto", soft=0.3)
    assert any("0.25" in str(w.message) for w in rec)
    # NTS SOFT=1 warns above 1
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn="auto", soft=1.5)
    assert any("unstable" in str(w.message) for w in rec)
    # a safe SOFSCL is silent
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn="auto", soft=0.1)
    assert not rec


def test_def_visc_excludes_tie():
    with pytest.raises(ValueError, match="visc.*tie|tie"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n=1e7, tie=True,
                   outward=(0.0, 0.0, 1.0), visc=0.5)


def test_def_visc_rejects_negative_accepts_zero():
    with pytest.raises(ValueError, match="visc"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn=1e6, visc=-1.0)
    # 0 is the fork's off-sentinel — accepted
    ContactDef(master_label="m", slave_label="s",
               formulation="nts", kn=1e6, visc=0.0)


def test_def_visc_zero_allowed_with_tie():
    # The fork refuses -visc with -tie only when the coefficient is active
    # (muc > 0.0 && isTie); visc=0 is the off-sentinel, so it must be accepted
    # with a tie (review #3/#4 — don't over-reject vs the fork).
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", eps_n=1e7, tie=True,
               outward=(0.0, 0.0, 1.0), visc=0.0)


def test_def_consistent_tan_accepted_both_lanes():
    ContactDef(master_label="m", slave_label="s",
               formulation="nts", kn=1e6, mu=0.3, consistent_tan=True)
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", eps_n=1e7, mu=0.3, consistent_tan=True)


# --------------------------------------------------------------------------
# ContactDef validation — edge-edge fallback (ADR-57 E2–E7)
# --------------------------------------------------------------------------
def test_def_edge_edge_is_mortar_only():
    # the fork routes the edge-edge fallback off the mortar lane (-edgeedge
    # requires -mortar).
    with pytest.raises(ValueError, match="mortar-only"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn=1e6, edge_edge=True)
    # a stray edge_* param on the NTS lane also trips the mortar-only gate
    with pytest.raises(ValueError, match="mortar-only"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="nts", kn=1e6, edge_mu=0.3)
    # mortar accepts it
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", eps_n=1e7, edge_edge=True)


@pytest.mark.parametrize("kw", [
    dict(edge_kn=1e7), dict(edge_band=0.01), dict(edge_mu=0.3),
    dict(edge_kt=1e6), dict(edge_cohesion=1e3), dict(edge_tau_max=5e5),
    dict(edge_consistent_tan=True), dict(edge_soft=True), dict(edge_alm=True),
    dict(edge_aug_tol=1e-6),
])
def test_def_edge_params_require_edge_edge(kw):
    # every edge_* knob requires edge_edge=True (the fork silently ignores
    # -edge* without -edgeedge; apeGmsh fails loud).
    with pytest.raises(ValueError, match="require edge_edge=True|edge_edge"):
        ContactDef(master_label="m", slave_label="s",
                   formulation="mortar", eps_n=1e7, **kw)
    # with edge_edge=True the same knob is accepted
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", eps_n=1e7, edge_edge=True, **kw)


def test_def_edge_kn_auto_or_positive():
    # "auto" is a valid sentinel; a typo / negative / zero is rejected.
    ContactDef(master_label="m", slave_label="s", formulation="mortar",
               eps_n=1e7, edge_edge=True, edge_kn="auto")
    with pytest.raises(ValueError, match="edge_kn"):
        ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n=1e7, edge_edge=True, edge_kn=-1.0)
    with pytest.raises(ValueError, match="number or 'auto'"):
        ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n=1e7, edge_edge=True, edge_kn="AUTO")


@pytest.mark.parametrize("kw, match", [
    (dict(edge_band=-1.0), "edge_band"),
    (dict(edge_band=0.0), "edge_band"),     # a band of 0 deactivates → reject
    (dict(edge_mu=-0.1), "edge_mu"),
    (dict(edge_kt=-1.0), "edge_kt"),
    (dict(edge_cohesion=-1.0), "edge_cohesion"),
    (dict(edge_tau_max=-1.0), "edge_tau_max"),
    (dict(edge_aug_tol=0.0), "edge_aug_tol"),
])
def test_def_edge_range_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n=1e7, edge_edge=True, **kw)


@pytest.mark.parametrize("kw", [
    dict(edge_mu=0.0), dict(edge_kt=0.0), dict(edge_cohesion=0.0),
    dict(edge_tau_max=0.0),     # tau_max=0 ⇒ "no cap" sentinel
])
def test_def_edge_friction_accepts_zero_sentinels(kw):
    ContactDef(master_label="m", slave_label="s", formulation="mortar",
               eps_n=1e7, edge_edge=True, **kw)


@pytest.mark.parametrize("bad", [0.0, -0.5, float("nan"), float("inf")])
def test_def_edge_soft_numeric_must_be_finite_positive(bad):
    with pytest.raises(ValueError, match="SOFSCL|finite"):
        ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n=1e7, edge_edge=True, edge_soft=bad)


def test_def_edge_soft_sofscl_warns_above_one():
    import warnings as _w
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n="auto", edge_edge=True, edge_soft=1.5)
    assert any("unstable" in str(w.message) for w in rec)
    # a safe SOFSCL is silent
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n="auto", edge_edge=True, edge_soft=0.1)
    assert not rec


def test_def_edge_edge_bare_enables_fallback():
    # edge_edge=True alone (no edge_* knobs) is valid — the fork sizes every
    # edge knob from defaults (penalty ⇒ mortar epsN, band ⇒ facet edge).
    d = ContactDef(master_label="m", slave_label="s", formulation="mortar",
                   eps_n="auto", edge_edge=True)
    assert d.edge_edge is True and d.edge_kn is None


# --------------------------------------------------------------------------
# Contact + handler-requiring MP guard predicate (#7)
# --------------------------------------------------------------------------
def test_handler_requiring_mp_predicate():
    from types import SimpleNamespace as NS

    from apeGmsh._kernel.records._kinds import ConstraintKind as K
    from apeGmsh.opensees.apesees import _fem_has_handler_requiring_mp

    def fem(recs):
        return NS(nodes=NS(constraints=recs))

    # True MP_Constraints (need a handler LadrunoContact can't provide)
    assert _fem_has_handler_requiring_mp(fem([NS(kind=K.EQUAL_DOF)]))
    assert _fem_has_handler_requiring_mp(fem([NS(kind=K.RIGID_DIAPHRAGM)]))
    assert _fem_has_handler_requiring_mp(
        fem([NS(kind=K.RIGID_BODY, as_element=False)]))
    # Handler-independent elements — must NOT trip the guard
    assert not _fem_has_handler_requiring_mp(fem([NS(kind=K.KINEMATIC_COUPLING)]))
    assert not _fem_has_handler_requiring_mp(
        fem([NS(kind=K.RIGID_BODY, as_element=True)]))
    # g.constraints.penalty → a stiff spring element, NOT an MP_Constraint:
    # must NOT trip the guard (else contact + penalty wrongly fails loud).
    assert not _fem_has_handler_requiring_mp(fem([NS(kind=K.PENALTY)]))
    assert not _fem_has_handler_requiring_mp(fem([]))


# --------------------------------------------------------------------------
# H5 deck-zone contact behavior (silent no-op + name-consume + handler skip)
# --------------------------------------------------------------------------
def test_h5_deck_contact_behavior():
    # The OpenSees *deck* zone no longer warns on contact: the NEUTRAL zone
    # now persists every ContactRecord (schema 2.21.0), so the deck-zone no-op
    # is silent (mirroring reinforce ties). The deck handler skip + name-consume
    # invariants still hold.
    import warnings as _w

    from apeGmsh.opensees.emitter.h5 import (
        H5Emitter,
        H5FeatureDeferredWarning,
        H5ReinforceDeviationWarning,
    )
    # back-compat alias is the SAME class
    assert H5ReinforceDeviationWarning is H5FeatureDeferredWarning

    e = H5Emitter(model_name="contact_deck")
    e.model(ndm=3, ndf=3)
    e.mp_constraint_comment("wall_tie")          # latch a declaration name
    with _w.catch_warnings(record=True) as rec:
        _w.simplefilter("always")
        e.contact_surface(1, "-master", 3, 1, 2, 3)
        e.contact_surface(2, "-slave", 4, 5)
        e.contact(1, 1, 2, 1.0e6)
    # contacts now round-trip via the neutral zone → no deferral warning
    deferred = [x for x in rec if issubclass(x.category, H5FeatureDeferredWarning)]
    assert len(deferred) == 0
    # the latched name was consumed (cannot leak onto the next real MP record)
    assert e._pending_mp_name == ""
    # the auto-emitted LadrunoContact handler is still NOT recorded in the deck
    # zone (which carries no contact deck records — the full model round-trips
    # via the neutral zone), so the replayed deck falls back to the default.
    e.constraints("LadrunoContact")
    assert e._chain_attrs.get("handler") != "LadrunoContact"
    # a real handler IS still recorded
    e.constraints("Transformation")
    assert e._chain_attrs.get("handler") == "Transformation"


# --------------------------------------------------------------------------
# Higher-order surface → corner-facet drop (ConstraintsComposite)
# --------------------------------------------------------------------------
@pytest.mark.parametrize("full_npe, expect_nps", [
    (3, 3),   # tri3
    (6, 3),   # tri6  → corner tri3
    (4, 4),   # quad4
    (8, 4),   # quad8 → corner quad4
    (9, 4),   # quad9 → corner quad4
])
def test_drop_to_corner_facets(full_npe, expect_nps):
    from apeGmsh.core.ConstraintsComposite import _drop_to_corner_facets
    faces = np.arange(2 * full_npe).reshape(2, full_npe)
    out, nps = _drop_to_corner_facets(faces)
    assert nps == expect_nps
    assert out.shape == (2, expect_nps)
    # corners are the LEADING columns (gmsh orders corners first)
    np.testing.assert_array_equal(out, faces[:, :expect_nps])


def test_drop_to_corner_facets_rejects_unsupported_width():
    from apeGmsh.core.ConstraintsComposite import _drop_to_corner_facets
    with pytest.raises(ValueError, match="not a supported tri"):
        _drop_to_corner_facets(np.arange(10).reshape(2, 5))


# --------------------------------------------------------------------------
# Record → emit (emit_contacts)
# --------------------------------------------------------------------------
class _Fem:
    def __init__(self, contacts):
        self.elements = type("E", (), {"contacts": contacts})()


def _nts_rec(**over):
    base = dict(
        kind="contact", formulation="nts",
        master_faces=np.array([[1, 2, 3], [3, 4, 1]]), master_nps=3,
        slave_nodes=[10, 11], kn=1.0e6, mu=0.3, kt=5.0e5,
        outward=(0.0, 0.0, 1.0),
    )
    base.update(over)
    return ContactRecord(**base)


def _mortar_rec(**over):
    base = dict(
        kind="contact", formulation="mortar",
        master_faces=np.array([[1, 2, 3]]), master_nps=3,
        slave_faces=np.array([[4, 5, 6]]), slave_nps=3,
        eps_n="auto", aug_tol=1e-8, max_aug=20, ngp=2,
    )
    base.update(over)
    return ContactRecord(**base)


def test_emit_nts_two_surfaces_and_contact():
    em = RecordingEmitter()
    emit_contacts(em, _Fem([_nts_rec()]), TagAllocator())
    surf = [c for c in em.calls if c[0] == "contact_surface"]
    con = [c for c in em.calls if c[0] == "contact"]
    assert len(surf) == 2 and len(con) == 1
    # master faceted (-master 3 flat…), slave node-set (-slave …)
    assert "-master" in surf[0][1] and "-slave" in surf[1][1]
    # contact verb carries kn kt mu + outward
    cargs = con[0][1]
    assert "-outward" in cargs and 1.0e6 in cargs


def test_emit_mortar_slave_segments():
    em = RecordingEmitter()
    emit_contacts(em, _Fem([_mortar_rec()]), TagAllocator())
    surf = [c for c in em.calls if c[0] == "contact_surface"]
    con = [c for c in em.calls if c[0] == "contact"][0][1]
    assert "-slave-segments" in surf[1][1]
    assert "-mortar" in con and "-epsN" in con


def test_emit_surface_and_contact_tags_distinct_namespaces():
    em = RecordingEmitter()
    emit_contacts(em, _Fem([_nts_rec()]), TagAllocator())
    surf = [c for c in em.calls if c[0] == "contact_surface"]
    con = [c for c in em.calls if c[0] == "contact"][0][1]
    m_tag, s_tag = surf[0][1][0], surf[1][1][0]
    c_tag = con[0]
    # two surface tags (1,2 in the contactSurface namespace), one contact tag
    assert {m_tag, s_tag} == {1, 2}
    assert c_tag == 1  # separate "contact" namespace starts at 1
    # the contact verb references the two surface tags
    assert con[1] == m_tag and con[2] == s_tag


def test_emit_noop_when_no_contacts():
    em = RecordingEmitter()
    emit_contacts(em, _Fem([]), TagAllocator())
    assert [c for c in em.calls if c[0] in ("contact", "contact_surface")] == []


def test_emit_carries_extension_modifiers():
    # a record with the extension fields flows them through to the contact verb.
    em = RecordingEmitter()
    rec = _nts_rec(kn="auto", kt=None, mu=None, outward=None,
                   soft=0.1, visc=1.0, consistent_tan=True, geom_tan=True)
    emit_contacts(em, _Fem([rec]), TagAllocator())
    cargs = [c for c in em.calls if c[0] == "contact"][0][1]
    for tok in ("-soft", 0.1, "-visc", 1.0, "-consistanttan", "-geomtan"):
        assert tok in cargs


def test_emit_carries_edge_edge_modifiers():
    # a mortar record with the edge-edge fields flows them through to the verb.
    em = RecordingEmitter()
    rec = _mortar_rec(
        aug_tol=None, max_aug=None, ngp=None,
        edge_edge=True, edge_kn="auto", edge_band=0.01, edge_mu=0.4,
        edge_kt=1e6, edge_cohesion=1e3, edge_tau_max=5e5,
        edge_consistent_tan=True, edge_soft=0.1, edge_alm=True,
        edge_aug_tol=1e-6)
    emit_contacts(em, _Fem([rec]), TagAllocator())
    cargs = [c for c in em.calls if c[0] == "contact"][0][1]
    for tok in ("-edgeedge", "-edgeKn", "auto", "-edgeBand", "-edgeMu",
                "-edgeKt", "-edgeCohesion", "-edgeTauMax",
                "-edgeConsistentTan", "-edgeSoft", "-edgeAlm", "-edgeAugTol"):
        assert tok in cargs, tok

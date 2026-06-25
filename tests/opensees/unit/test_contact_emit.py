"""Emit + validation contract for `g.constraints.contact` → fork contact.

Text-only (no OpenSees backend): locks the two grammar builders
(`contact_surface_args` / `contact_args`), the def validation (`ContactDef`),
and the record→emit path (`emit_contacts` → contact_surface/contact emitter
calls). Core-first scope (NTS penalty + mortar frictionless/friction +
explicit `-outward`; `-soft`/`-visc`/`-consistanttan`/`-geomtan` deferred).
The wrapper never auto-derives `-outward` — the fork derives a correct
per-facet normal otherwise (see ConstraintsComposite.resolve_contacts).
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
def test_nts_kn_only_one_number():
    a = contact_args(1, 2, "nts", kn=1.0e6)
    assert a == [1, 2, 1.0e6]


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
    (dict(formulation="nts", kn=0.0), "kn"),
    (dict(formulation="nts", kn=1e6, kt=-1.0), "kt"),
    (dict(formulation="nts", kn=1e6, mu=-0.1), "mu"),
    (dict(formulation="mortar", eps_n=-1.0), "eps_n"),
    (dict(formulation="mortar", eps_n=0.0), "eps_n"),
    (dict(formulation="mortar", eps_n=1e7, eps_t=-1.0), "eps_t"),
    (dict(formulation="mortar", eps_n=1e7, cohesion=-1.0), "cohesion"),
    (dict(formulation="mortar", eps_n=1e7, tau_max=0.0), "tau_max"),
    (dict(formulation="mortar", eps_n=1e7, aug_tol=0.0), "aug_tol"),
    (dict(formulation="mortar", eps_n=1e7, max_aug=0), "max_aug"),
    (dict(formulation="mortar", eps_n=1e7, ngp=0), "ngp"),
])
def test_def_range_validation(kw, match):
    with pytest.raises(ValueError, match=match):
        ContactDef(master_label="m", slave_label="s", **kw)


def test_def_auto_kn_skips_range_check():
    # "auto" is a valid sentinel and must not trip the numeric kn>0 check.
    ContactDef(master_label="m", slave_label="s",
               formulation="nts", kn="auto")
    ContactDef(master_label="m", slave_label="s",
               formulation="mortar", eps_n="auto")


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

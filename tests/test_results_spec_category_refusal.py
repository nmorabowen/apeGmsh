"""#2344: a recorder component in the wrong category refuses by name.

A canonical component under a category that does not record it was
accepted and recorded nothing, or another response under its name:

- ``elements`` + ``stress_xx`` emitted ``globalForce``;
- ``line_stations`` + ``displacement_x`` emitted ``section force``;
- ``gauss`` + ``displacement_x`` and ``nodes`` + ``stress_xx`` emitted no
  recorder on the ``.out`` route, with no warning.

The declaration (``opensees.recorder.RecorderRecord``) let any canonical
through in the element categories, and two hand-kept category tables
disagreed on 20 components. Now one table,
``apeGmsh._vocabulary.CATEGORY_COMPONENTS``, is what the declaration and the
resolved record read, and a
wrong-category component refuses at construction, naming its category.

Oracles:

- the four mis-routes above refuse by name (each passed on origin/main
  17af8a1c6);
- a full-table sweep: every pair the table holds emits byte-identical
  commands on every route (``.out`` Tcl / Python / live args, MPCO Tcl /
  Python / live args, the fork's ``ladruno`` args) to origin/main's emit.
  ``_EMIT_GOLDEN`` is the sha256 of :func:`emit_fingerprint`, generated
  with this file's helper on unmodified origin/main 17af8a1c6. Refusals
  and warnings are part of the fingerprint, so a route that refused a pair
  before still refuses it the same way.
"""
from __future__ import annotations

import hashlib
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest

from apeGmsh import _vocabulary as voc
from apeGmsh.opensees import recorder as decl
from apeGmsh.results.readers._ladruno_live import fork_recorder_args
from apeGmsh.results.spec import _emit
from apeGmsh.results.spec import _resolved
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord

#: The four mis-routes of #2344 and the category that records each.
MISROUTES = (
    ("elements", "stress_xx", "gauss"),
    ("line_stations", "displacement_x", "nodes"),
    ("gauss", "displacement_x", "nodes"),
    ("nodes", "stress_xx", "gauss"),
)

#: Indexed canonicals the sweep samples, per stem.
_INDEX_SAMPLES = ("0", "3")


def _resolved_record(category: str, components: Any) -> ResolvedRecorderRecord:
    ids = np.array([1, 2])
    return ResolvedRecorderRecord(
        category=category, name="r", components=components,
        dt=None, n_steps=None,
        node_ids=ids if category == "nodes" else None,
        element_ids=None if category in ("nodes", "modal") else ids,
        n_modes=2 if category == "modal" else None,
    )


def _decl_record(category: str, components: Any) -> decl.RecorderRecord:
    if category == "modal":
        return decl.RecorderRecord(
            category=category, components=components, n_modes=2)
    return decl.RecorderRecord(
        category=category, components=components, ids=(1, 2))


def _accepts(build: Callable[[], object]) -> bool:
    try:
        build()
    except ValueError:
        return False
    return True


def _table_pairs() -> list[tuple[str, str]]:
    """Every (category, component) the table records, indexed sampled."""
    pairs = []
    for category, comps in voc.CATEGORY_COMPONENTS.items():
        pairs += [(category, c) for c in comps]
        pairs += [
            (category, stem + n)
            for stem in voc.CATEGORY_INDEXED_STEMS[category]
            for n in _INDEX_SAMPLES
        ]
    return sorted(pairs)


def _every_candidate() -> list[str]:
    """Every canonical, plus the indexed samples of every stem."""
    stems = {s for v in voc.CATEGORY_INDEXED_STEMS.values() for s in v}
    return sorted(
        set(voc.ALL_CANONICAL)
        | {s + n for s in stems for n in _INDEX_SAMPLES}
    )


# ---------------------------------------------------------------------
# The four mis-routes refuse by name
# ---------------------------------------------------------------------

@pytest.mark.parametrize("category,component,right", MISROUTES)
def test_resolved_record_refuses_wrong_category(
    category: str, component: str, right: str,
) -> None:
    with pytest.raises(ValueError) as info:
        _resolved_record(category, (component,))
    msg = str(info.value)
    assert repr(component) in msg
    assert f"belongs to category {right!r}" in msg
    assert f"not {category!r}" in msg


@pytest.mark.parametrize("category,component,right", MISROUTES)
def test_declaration_refuses_wrong_category(
    category: str, component: str, right: str,
) -> None:
    with pytest.raises(ValueError) as info:
        _decl_record(category, (component,))
    assert f"belongs to category {right!r}" in str(info.value)


@pytest.mark.parametrize("category,component,right", MISROUTES)
def test_declare_kwargs_refuse_wrong_category(
    category: str, component: str, right: str,
) -> None:
    """The ``ops.recorder.declare`` builder refuses too, before any emit."""
    with pytest.raises(ValueError, match=f"belongs to category {right!r}"):
        decl.build_recorder_declaration(
            ndm=3, ndf=6, pg="Top", **{category: component})


@pytest.mark.parametrize("category,component", [
    ("gauss", "spring_force_0"),
    ("gauss", "fiber_stress_1"),
    ("elements", "state_variable_1"),
    ("line_stations", "spring_force"),
    ("modal", "displacement_x"),
])
def test_indexed_and_modal_wrong_category_refuse(
    category: str, component: str,
) -> None:
    for build in (_resolved_record, _decl_record):
        with pytest.raises(ValueError, match="belongs to category"):
            build(category, (component,))


def test_names_every_category_of_a_shared_component() -> None:
    with pytest.raises(ValueError) as info:
        _resolved_record("nodes", ("axial_force",))
    assert "belongs to category 'line_stations' or 'gauss'" in str(info.value)


def test_resolved_record_refuses_unknown_category() -> None:
    with pytest.raises(ValueError, match="unknown category 'bogus'"):
        _resolved_record("bogus", ("displacement_x",))


@pytest.mark.parametrize("build", [_resolved_record, _decl_record])
def test_bare_string_is_one_component(build: Callable[..., Any]) -> None:
    """A bare string was iterated, and the refusal named its letter 'd'."""
    rec = build("nodes", "displacement_x")
    assert rec.components == ("displacement_x",)


# ---------------------------------------------------------------------
# One table
# ---------------------------------------------------------------------

def test_categories_agree_everywhere() -> None:
    cats = tuple(voc.CATEGORY_COMPONENTS)
    assert cats == tuple(voc.CATEGORY_INDEXED_STEMS)
    assert _resolved.ALL_CATEGORIES == cats
    assert decl.ALL_RECORDER_CATEGORIES == cats


def test_table_holds_canonical_names_and_covers_every_canonical() -> None:
    held = set().union(*voc.CATEGORY_COMPONENTS.values())
    assert held == set(voc.ALL_CANONICAL)
    for stems in voc.CATEGORY_INDEXED_STEMS.values():
        for stem in stems:
            assert voc.is_canonical(stem + "0"), stem


@pytest.mark.parametrize("component", _every_candidate())
def test_every_component_agrees_on_every_path(component: str) -> None:
    """Declaration, resolved record and the table answer alike."""
    for category in voc.CATEGORY_COMPONENTS:
        want = voc.in_category(category, component)
        assert _accepts(lambda: _resolved_record(category, (component,))) \
            is want, (category, component)
        assert _accepts(lambda: _decl_record(category, (component,))) \
            is want, (category, component)
    assert voc.categories_of(component) == tuple(
        c for c in voc.CATEGORY_COMPONENTS if voc.in_category(c, component))


# ---------------------------------------------------------------------
# Full-table sweep: valid pairs emit what origin/main emitted
# ---------------------------------------------------------------------

#: Element classes each category's records are emitted under, besides none.
_SWEEP_CLASSES: dict[str, tuple[str, ...]] = {
    "nodes": (),
    "elements": ("ZeroLength", "elasticBeamColumn"),
    "line_stations": ("ForceBeamColumn3d", "elasticBeamColumn"),
    "gauss": ("FourNodeTetrahedron", "ShellMITC4", "Truss"),
    "fibers": ("ForceBeamColumn3d",),
    "layers": ("ShellMITC4",),
}


def _outcome(fn: Callable[[], Any]) -> str:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            got = fn()
            text = "\n".join(got) if isinstance(got, list) else repr(got)
        except Exception as exc:  # the golden pins refusals too
            text = f"RAISE {type(exc).__name__}: {exc}"
    for w in caught:
        text += f"\nWARN {w.category.__name__}: {w.message}"
    return text


def _routes(rec: ResolvedRecorderRecord) -> str:
    def logical() -> list:
        out = []
        for lr in _emit.emit_logical(rec):
            if lr.kind == "Node":  # slot DOFs as deck DOFs: a 3-D ndf-6 node
                lr = replace(lr, deck_dofs=lr.dofs)
            out += [_emit.format_tcl(lr), _emit.format_python(lr),
                    repr(_emit.to_ops_args(lr))]
        return out

    recs = (rec,)
    return "\n--\n".join([
        _outcome(logical),
        _outcome(lambda: _emit.emit_mpco_tcl(
            recs, output_dir="", filename="run.mpco")),
        _outcome(lambda: _emit.emit_mpco_python(
            recs, output_dir="", filename="run.mpco")),
        _outcome(lambda: _emit.mpco_ops_args(
            recs, output_dir="", filename="run.mpco")),
        _outcome(lambda: fork_recorder_args(recs, Path("run.ladruno"))),
    ])


def emit_fingerprint(category: str, component: str) -> str:
    """sha256 of every route's emitted command (or refusal) for one pair."""
    parts = []
    for cls in (None, *_SWEEP_CLASSES[category]):
        ids = np.array([1, 2])
        rec = ResolvedRecorderRecord(
            category=category, name="r", components=(component,),
            dt=None if cls is None else 0.5, n_steps=None,
            node_ids=ids if category == "nodes" else None,
            element_ids=None if category == "nodes" else ids,
            element_class_name=cls,
        )
        parts.append(f"## {cls}\n{_routes(rec)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:20]


def test_golden_covers_exactly_the_table() -> None:
    """A component added to the table needs its emit pinned here."""
    pairs = {f"{c}|{k}" for c, k in _table_pairs() if c != "modal"}
    assert set(_EMIT_GOLDEN) == pairs


@pytest.mark.parametrize(
    "category,component",
    [(c, k) for c, k in _table_pairs() if c != "modal"],
)
def test_valid_pair_emits_what_main_emitted(
    category: str, component: str,
) -> None:
    assert emit_fingerprint(category, component) == \
        _EMIT_GOLDEN[f"{category}|{component}"]



#: sha256[:20] of emit_fingerprint per pair, on origin/main 17af8a1c6.
_EMIT_GOLDEN: dict[str, str] = {
    "elements|nodal_resisting_force_local_x": "db4912bd139443830e7c",
    "elements|nodal_resisting_force_local_y": "db4912bd139443830e7c",
    "elements|nodal_resisting_force_local_z": "db4912bd139443830e7c",
    "elements|nodal_resisting_force_x": "fd7c7c17611f86968c95",
    "elements|nodal_resisting_force_y": "fd7c7c17611f86968c95",
    "elements|nodal_resisting_force_z": "fd7c7c17611f86968c95",
    "elements|nodal_resisting_moment_local_x": "db4912bd139443830e7c",
    "elements|nodal_resisting_moment_local_y": "db4912bd139443830e7c",
    "elements|nodal_resisting_moment_local_z": "db4912bd139443830e7c",
    "elements|nodal_resisting_moment_x": "fd7c7c17611f86968c95",
    "elements|nodal_resisting_moment_y": "fd7c7c17611f86968c95",
    "elements|nodal_resisting_moment_z": "fd7c7c17611f86968c95",
    "elements|spring_deformation": "795b8bb309a0f1292350",
    "elements|spring_deformation_0": "795b8bb309a0f1292350",
    "elements|spring_deformation_3": "795b8bb309a0f1292350",
    "elements|spring_force": "3ed6762734c10672219e",
    "elements|spring_force_0": "3ed6762734c10672219e",
    "elements|spring_force_3": "3ed6762734c10672219e",
    "fibers|damage": "24e2e24ea8a45b57ac6e",
    "fibers|damage_compression": "830de79df5d1c823c969",
    "fibers|damage_tension": "055bf8bfa5e975eaaabf",
    "fibers|equivalent_plastic_strain_compression": "d291a427399896fc8ca9",
    "fibers|equivalent_plastic_strain_tension": "b5895b1ca0a50bdf7000",
    "fibers|fiber_strain": "abe65bbe93aedaa5f796",
    "fibers|fiber_strain_0": "eca2467abdea1d84602b",
    "fibers|fiber_strain_3": "aac8b29d1a16eb0302d9",
    "fibers|fiber_stress": "55de193aad956a0bd988",
    "fibers|fiber_stress_0": "afa11969fd164f6ddc50",
    "fibers|fiber_stress_3": "688bf9b083744accb87f",
    "fibers|state_variable_0": "91f4a3cc48adc17c7d27",
    "fibers|state_variable_3": "788704f17138d5371ba0",
    "gauss|axial_force": "0dc471252688eb8116a8",
    "gauss|bending_moment_xx": "142926fa15d44f143862",
    "gauss|bending_moment_xy": "142926fa15d44f143862",
    "gauss|bending_moment_yy": "142926fa15d44f143862",
    "gauss|curvature_xx": "ee3e1079825ee3b3764a",
    "gauss|curvature_xy": "ee3e1079825ee3b3764a",
    "gauss|curvature_yy": "ee3e1079825ee3b3764a",
    "gauss|damage": "add4e673c3c1e7badf03",
    "gauss|damage_compression": "add4e673c3c1e7badf03",
    "gauss|damage_tension": "add4e673c3c1e7badf03",
    "gauss|equivalent_plastic_strain": "6518768b1177420cba27",
    "gauss|equivalent_plastic_strain_compression": "76eb5c719f8cf48f0a5b",
    "gauss|equivalent_plastic_strain_current": "2cf5625d18232aa5850b",
    "gauss|equivalent_plastic_strain_tension": "76eb5c719f8cf48f0a5b",
    "gauss|j2_plastic_strain": "2cf5625d18232aa5850b",
    "gauss|j2_strain": "d113d1253da1a3fd5889",
    "gauss|j2_stress": "b0b716454cfff0bdbbcf",
    "gauss|j3_stress": "b0b716454cfff0bdbbcf",
    "gauss|lode_angle": "b0b716454cfff0bdbbcf",
    "gauss|max_shear_plastic_strain": "2cf5625d18232aa5850b",
    "gauss|max_shear_strain": "d113d1253da1a3fd5889",
    "gauss|max_shear_stress": "b0b716454cfff0bdbbcf",
    "gauss|mean_stress": "b0b716454cfff0bdbbcf",
    "gauss|membrane_force_xx": "142926fa15d44f143862",
    "gauss|membrane_force_xy": "142926fa15d44f143862",
    "gauss|membrane_force_yy": "142926fa15d44f143862",
    "gauss|membrane_strain_xx": "ee3e1079825ee3b3764a",
    "gauss|membrane_strain_xy": "ee3e1079825ee3b3764a",
    "gauss|membrane_strain_yy": "ee3e1079825ee3b3764a",
    "gauss|plastic_strain_xx": "92c5c1eeb72eb45ce0ed",
    "gauss|plastic_strain_xy": "92c5c1eeb72eb45ce0ed",
    "gauss|plastic_strain_xz": "92c5c1eeb72eb45ce0ed",
    "gauss|plastic_strain_yy": "92c5c1eeb72eb45ce0ed",
    "gauss|plastic_strain_yz": "92c5c1eeb72eb45ce0ed",
    "gauss|plastic_strain_zz": "92c5c1eeb72eb45ce0ed",
    "gauss|pressure_hydrostatic": "b0b716454cfff0bdbbcf",
    "gauss|principal_plastic_strain_1": "2cf5625d18232aa5850b",
    "gauss|principal_plastic_strain_2": "2cf5625d18232aa5850b",
    "gauss|principal_plastic_strain_3": "2cf5625d18232aa5850b",
    "gauss|principal_strain_1": "d113d1253da1a3fd5889",
    "gauss|principal_strain_2": "d113d1253da1a3fd5889",
    "gauss|principal_strain_3": "d113d1253da1a3fd5889",
    "gauss|principal_stress_1": "b0b716454cfff0bdbbcf",
    "gauss|principal_stress_2": "b0b716454cfff0bdbbcf",
    "gauss|principal_stress_3": "b0b716454cfff0bdbbcf",
    "gauss|state_variable_0": "a3dce1c0f1e226397865",
    "gauss|state_variable_3": "14182deef2699028af79",
    "gauss|strain_xx": "ee3e1079825ee3b3764a",
    "gauss|strain_xy": "ee3e1079825ee3b3764a",
    "gauss|strain_xz": "ee3e1079825ee3b3764a",
    "gauss|strain_yy": "ee3e1079825ee3b3764a",
    "gauss|strain_yz": "ee3e1079825ee3b3764a",
    "gauss|strain_zz": "ee3e1079825ee3b3764a",
    "gauss|stress_triaxiality": "b0b716454cfff0bdbbcf",
    "gauss|stress_xx": "142926fa15d44f143862",
    "gauss|stress_xy": "142926fa15d44f143862",
    "gauss|stress_xz": "142926fa15d44f143862",
    "gauss|stress_yy": "142926fa15d44f143862",
    "gauss|stress_yz": "142926fa15d44f143862",
    "gauss|stress_zz": "142926fa15d44f143862",
    "gauss|transverse_shear_strain_xz": "ee3e1079825ee3b3764a",
    "gauss|transverse_shear_strain_yz": "ee3e1079825ee3b3764a",
    "gauss|transverse_shear_xz": "142926fa15d44f143862",
    "gauss|transverse_shear_yz": "142926fa15d44f143862",
    "gauss|tresca_stress": "b0b716454cfff0bdbbcf",
    "gauss|volumetric_plastic_strain": "2cf5625d18232aa5850b",
    "gauss|volumetric_strain": "d113d1253da1a3fd5889",
    "gauss|von_mises_shell": "b0b716454cfff0bdbbcf",
    "gauss|von_mises_strain": "d113d1253da1a3fd5889",
    "gauss|von_mises_stress": "b0b716454cfff0bdbbcf",
    "layers|damage": "439c76f09cb6cdf526a3",
    "layers|damage_compression": "439c76f09cb6cdf526a3",
    "layers|damage_tension": "439c76f09cb6cdf526a3",
    "layers|equivalent_plastic_strain_compression": "a67e4466a243e5a0bcd3",
    "layers|equivalent_plastic_strain_tension": "a67e4466a243e5a0bcd3",
    "layers|fiber_strain": "9cff333a94f12fa2f9db",
    "layers|fiber_strain_0": "9cff333a94f12fa2f9db",
    "layers|fiber_strain_3": "9cff333a94f12fa2f9db",
    "layers|fiber_stress": "947c3c4d3b93a825cdc6",
    "layers|fiber_stress_0": "947c3c4d3b93a825cdc6",
    "layers|fiber_stress_3": "947c3c4d3b93a825cdc6",
    "layers|state_variable_0": "439c76f09cb6cdf526a3",
    "layers|state_variable_3": "439c76f09cb6cdf526a3",
    "line_stations|axial_force": "88353e0f3bd4344df682",
    "line_stations|axial_strain": "0dfb700eedd3117134e5",
    "line_stations|bending_moment_y": "88353e0f3bd4344df682",
    "line_stations|bending_moment_z": "88353e0f3bd4344df682",
    "line_stations|curvature_y": "fc50eaa8c88fdec90fa0",
    "line_stations|curvature_z": "46b16e19e7ce7a80f954",
    "line_stations|shear_strain_y": "d97a05b6291ec36f86e2",
    "line_stations|shear_strain_z": "4180af243b917ad9279c",
    "line_stations|shear_y": "88353e0f3bd4344df682",
    "line_stations|shear_z": "88353e0f3bd4344df682",
    "line_stations|torsion": "88353e0f3bd4344df682",
    "line_stations|torsional_strain": "3552b0dd0eb5576b4f16",
    "nodes|acceleration_x": "4b970fd1dc120ae55d64",
    "nodes|acceleration_y": "f510b2cce9a301cc4e3a",
    "nodes|acceleration_z": "b51e5a6eef05b4b7627b",
    "nodes|angular_acceleration_x": "4162eb0af826f3a71b72",
    "nodes|angular_acceleration_y": "26573c237cf6aef2228f",
    "nodes|angular_acceleration_z": "bdc1ce98c87719a79b09",
    "nodes|angular_velocity_x": "f14fe5f9b1ab07835af6",
    "nodes|angular_velocity_y": "472c4e496cabf2a2ac7e",
    "nodes|angular_velocity_z": "ef90033ede581bd64f7c",
    "nodes|displacement_increment_x": "b08f6212687b1d069596",
    "nodes|displacement_increment_y": "0ebdb65c8bc96b3045a6",
    "nodes|displacement_increment_z": "5afd18117e8da988a3c8",
    "nodes|displacement_x": "8bae911826a0dc95834f",
    "nodes|displacement_y": "623e1310fe3e961402b1",
    "nodes|displacement_z": "c3f4ba2ea76d6b3d3ed7",
    "nodes|force_x": "279ab49fd6cfde0a200e",
    "nodes|force_y": "5b9808092a885cf85eca",
    "nodes|force_z": "96e38d5bb1859099a59c",
    "nodes|moment_x": "47c82d6188770d2eb316",
    "nodes|moment_y": "cd21d0c6c040a8f92361",
    "nodes|moment_z": "89d913ba2b1442480187",
    "nodes|pore_pressure": "2cde03b568000399e222",
    "nodes|pore_pressure_rate": "36cac2d4420599e645b9",
    "nodes|reaction_force_x": "e6b5c543497a9e29873a",
    "nodes|reaction_force_y": "c8b7d16c34d0bdd695c3",
    "nodes|reaction_force_z": "6b8e4159cefeab867547",
    "nodes|reaction_moment_x": "56e93ac108ae8fa78d37",
    "nodes|reaction_moment_y": "ec1a0874c1f7720c1f1f",
    "nodes|reaction_moment_z": "f7f0349a4dbaad09556b",
    "nodes|rotation_x": "9a1ebb56a9b42aece877",
    "nodes|rotation_y": "538e3d0ed4db3b4ce19b",
    "nodes|rotation_z": "2a8bbaa0a3e83cd57dd3",
    "nodes|velocity_x": "534e890757c1ab4e331e",
    "nodes|velocity_y": "eacbb5064ec2a07e30b9",
    "nodes|velocity_z": "336bc1615c10a641bcf8",
}

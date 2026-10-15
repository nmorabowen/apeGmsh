"""Live-emit argument tuples — ``to_ops_args`` and ``mpco_ops_args``.

These tests cover the new helpers that produce ready-to-splat
argument tuples for ``ops.recorder(*args)`` calls. They are the
foundation for the live recorder strategy
(``ResolvedRecorderSpec.emit_recorders`` / ``emit_mpco``).

The tests do not import openseespy — they verify the tuples have
the right shape, types, and ordering, and that they agree with the
existing source-code formatters where they overlap.
"""
from __future__ import annotations

import numpy as np
import pytest

from apeGmsh._vocabulary import (
    DERIVED_PLASTIC_STRAIN_SCALARS,
    DERIVED_SCALARS,
    LINE_DIAGRAMS,
    LINE_STATION_DEFORMATIONS,
    MATERIAL_STATE,
    PLASTIC_STRAIN,
    STRAIN,
    STRESS,
    expand_shorthand,
)
from apeGmsh.opensees._response_catalog import (
    RESPONSE_CATALOG,
    gauss_keyword_for_canonical,
    gauss_routing_for_canonical,
    mpco_gauss_group_aliases,
)
from apeGmsh.results.readers._mpco_line_io import (
    MPCO_SECTION_NAME_TO_CODE,
    canonical_to_line_station_token,
)
from apeGmsh.results.readers._mpco_material_io import parent_token_for_canonical
from apeGmsh.results.spec._emit import (
    collect_mpco_tokens,
    emit_logical,
    emit_mpco_python,
    emit_mpco_tcl,
    format_python,
    mpco_ops_args,
    to_deck,
    to_ops_args,
)
from apeGmsh.results.spec._resolved import ResolvedRecorderRecord
from tests.fixtures.recorder_models import node_model


def _node_logicals(rec: ResolvedRecorderRecord, **kw):
    """A node record's recorders as they reach OpenSees: DOFs mapped by
    a model of 3-D frame nodes (``to_deck``; #2018)."""
    model = node_model(rec.node_ids, "h")
    return [to_deck(lr, None, model, where="t")
            for lr in emit_logical(rec, **kw)]


# =====================================================================
# to_ops_args — Node recorder
# =====================================================================

def test_to_ops_args_node_displacement_basic() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None,
        node_ids=np.array([1, 2]),
    )
    [logical] = _node_logicals(rec)
    args = to_ops_args(logical)

    # Shape: ('Node', '-file', '<path>', '-time', '-node', 1, 2,
    #         '-dof', 1, 2, 'disp')
    assert args[0] == "Node"
    assert args[1] == "-file"
    assert args[2] == "r_disp.out"
    assert args[3] == "-time"
    assert args[4] == "-node"
    # Node IDs are int, not str
    assert args[5] == 1 and isinstance(args[5], int)
    assert args[6] == 2 and isinstance(args[6], int)
    assert args[7] == "-dof"
    assert args[8] == 1 and isinstance(args[8], int)
    assert args[9] == 2 and isinstance(args[9], int)
    assert args[-1] == "disp"


def test_to_ops_args_node_with_dt() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([5]),
    )
    [logical] = _node_logicals(rec)
    args = to_ops_args(logical)

    # -dT must be present, value must be float
    assert "-dT" in args
    dt_idx = args.index("-dT")
    assert args[dt_idx + 1] == 0.01
    assert isinstance(args[dt_idx + 1], float)


def test_to_ops_args_node_xml_format() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    [logical] = _node_logicals(rec, file_format="xml")
    args = to_ops_args(logical)

    assert "-xml" in args
    assert "-file" not in args
    xml_idx = args.index("-xml")
    assert args[xml_idx + 1] == "r_disp.xml"


def test_to_ops_args_node_with_output_dir() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    [logical] = _node_logicals(rec, output_dir="out/")
    args = to_ops_args(logical)

    file_idx = args.index("-file")
    assert args[file_idx + 1] == "out/r_disp.out"


# =====================================================================
# to_ops_args — Element recorder
# =====================================================================

def test_to_ops_args_element_gauss_stress() -> None:
    rec = ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([100, 101]),
    )
    [logical] = emit_logical(rec)
    args = to_ops_args(logical)

    assert args[0] == "Element"
    assert "-ele" in args
    ele_idx = args.index("-ele")
    assert args[ele_idx + 1] == 100 and isinstance(args[ele_idx + 1], int)
    assert args[ele_idx + 2] == 101
    # No -dof for element recorders
    assert "-dof" not in args
    # Last token is the gauss keyword
    assert args[-1] == "stresses"


def test_to_ops_args_element_line_stations_pair() -> None:
    """line_stations records produce two logical recorders — section
    force + integrationPoints. Both should round-trip through
    to_ops_args cleanly."""
    rec = ResolvedRecorderRecord(
        category="line_stations", name="beam",
        components=("axial_force",),
        dt=None, n_steps=None,
        element_ids=np.array([1, 2, 3]),
    )
    logicals = list(emit_logical(rec))
    assert len(logicals) == 2

    args_force = to_ops_args(logicals[0])
    args_gpx = to_ops_args(logicals[1])

    # First recorder: section force tokens
    assert args_force[0] == "Element"
    assert "section" in args_force and "force" in args_force

    # Second recorder: integrationPoints
    assert args_gpx[0] == "Element"
    assert "integrationPoints" in args_gpx


# =====================================================================
# Cross-check: to_ops_args agrees with format_python
# =====================================================================

def test_to_ops_args_round_trip_with_format_python() -> None:
    """The arg tuple, when passed through repr-equivalent, should
    reproduce the format_python source exactly. Catches drift between
    the two."""
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "velocity_y"),
        dt=0.005, n_steps=None,
        node_ids=np.array([10, 20, 30]),
    )
    for logical in _node_logicals(rec):
        py_source = format_python(logical)
        args = to_ops_args(logical)

        # Every element of the tuple should appear in the source
        # (string args wrapped in quotes via repr; numbers as-is).
        for a in args:
            if isinstance(a, str):
                assert repr(a) in py_source, (
                    f"{a!r} not in {py_source!r}"
                )
            else:
                assert str(a) in py_source, (
                    f"{a} not in {py_source!r}"
                )


# =====================================================================
# mpco_ops_args
# =====================================================================

def test_mpco_ops_args_nodal_only() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x", "displacement_y"),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    args = mpco_ops_args([rec])

    # ('mpco', '<path>', '-N', 'displacement')
    assert args[0] == "mpco"
    assert args[1] == "run.mpco"
    assert args[2] == "-N"
    assert "displacement" in args
    assert "-E" not in args


def test_mpco_ops_args_element_only() -> None:
    rec = ResolvedRecorderRecord(
        category="gauss", name="r",
        components=("stress_xx",),
        dt=None, n_steps=None,
        element_ids=np.array([1]),
    )
    args = mpco_ops_args([rec])

    assert args[0] == "mpco"
    assert "-N" not in args
    assert "-E" in args
    assert "stresses" in args


def test_mpco_ops_args_mixed_with_dt() -> None:
    n_rec = ResolvedRecorderRecord(
        category="nodes", name="rn",
        components=("displacement_x",),
        dt=0.02, n_steps=None,
        node_ids=np.array([1]),
    )
    e_rec = ResolvedRecorderRecord(
        category="gauss", name="re",
        components=("strain_xx",),
        dt=0.01, n_steps=None,
        element_ids=np.array([1]),
    )
    with pytest.warns(UserWarning, match=r"takes one cadence.*-T dt 0\.01"):
        args = mpco_ops_args([n_rec, e_rec])

    # Both -N and -E
    assert "-N" in args
    assert "-E" in args
    # Cadence: smallest dt wins → 0.01
    assert "-T" in args
    t_idx = args.index("-T")
    assert args[t_idx + 1] == "dt"
    assert args[t_idx + 2] == 0.01
    assert isinstance(args[t_idx + 2], float)


def test_mpco_ops_args_output_dir_and_filename() -> None:
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=None, n_steps=None,
        node_ids=np.array([1]),
    )
    args = mpco_ops_args([rec], output_dir="results/", filename="my_run.mpco")
    assert args[1] == "results/my_run.mpco"


def test_mpco_ops_args_round_trip_with_emit_mpco_python() -> None:
    """Tuple agrees with the source-code form."""
    rec = ResolvedRecorderRecord(
        category="nodes", name="r",
        components=("displacement_x",),
        dt=0.01, n_steps=None,
        node_ids=np.array([1]),
    )
    py_source = emit_mpco_python([rec])
    args = mpco_ops_args([rec])

    for a in args:
        if isinstance(a, str):
            assert repr(a) in py_source
        else:
            assert str(a) in py_source


def test_mpco_ops_args_modal_record_emits_modes_token() -> None:
    rec = ResolvedRecorderRecord(
        category="modal", name="modes",
        components=(),
        dt=None, n_steps=None,
        n_modes=10,
    )
    args = mpco_ops_args([rec])
    # Modal records contribute the modes-of-vibration tokens on -N.
    assert "modesOfVibration" in args


# =====================================================================
# mpco_ops_args — gauss records route like the MPCO read side (#1991)
# =====================================================================
#
# Oracle: the MPCO reader finds a gauss-points canonical under the
# ``ON_ELEMENTS/<keyword>`` group that ``gauss_routing_for_canonical``
# names (``discover_gauss_buckets``), so the emit must write that group.
# Before #1991 a shell record emitted ``-E section.force`` (the beam
# bucket, reached through ``bending_moment``) and dropped
# ``membrane_force_*`` and ``transverse_shear_*`` silently: the MPCO
# gauss slab read empty.

#: Every catalogued shell class with its stress-resultant and
#: generalized-strain layouts, read from RESPONSE_CATALOG so a new shell
#: class is covered without editing this file.
_SHELL_LAYOUTS: dict[tuple[str, str], tuple[str, ...]] = {
    (cls, tok): tuple(layout.component_layout)
    for (cls, _rule, tok), layout in RESPONSE_CATALOG.items()
    if tok in ("stress", "strain")
    and tuple(layout.component_layout)[0] in ("membrane_force_xx",
                                              "membrane_strain_xx")
}
_SHELL_CLASSES = sorted({cls for cls, _ in _SHELL_LAYOUTS})
_MPCO_KEYWORD = {"stress": "stresses", "strain": "strains"}


def _gauss(components, cls=None, name="sh") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="gauss", name=name, components=tuple(components),
        dt=None, n_steps=None, element_ids=np.array([1]),
        element_class_name=cls,
    )


def test_shell_class_table_is_not_vacuous() -> None:
    assert {"ASDShellQ4", "ASDShellT3", "ShellMITC4", "ShellDKGQ",
            "ShellNLDKGQ", "ShellDKGT", "ShellNLDKGT"} <= set(_SHELL_CLASSES)
    for cls in _SHELL_CLASSES:
        assert (cls, "stress") in _SHELL_LAYOUTS
        assert (cls, "strain") in _SHELL_LAYOUTS


@pytest.mark.parametrize("cls", _SHELL_CLASSES)
@pytest.mark.parametrize("tok", ["stress", "strain"])
def test_mpco_shell_layout_emits_read_side_keyword(cls: str, tok: str) -> None:
    comps = _SHELL_LAYOUTS[(cls, tok)]
    args = mpco_ops_args([_gauss(comps, cls)])
    assert args == ("mpco", "run.mpco", "-E", _MPCO_KEYWORD[tok])
    # Cross-check against the reader's routing, component by component.
    for c in comps:
        routing = gauss_routing_for_canonical(c)
        assert routing is not None and routing[0] == _MPCO_KEYWORD[tok], c


@pytest.mark.parametrize("cls", _SHELL_CLASSES)
def test_mpco_shell_resultant_shorthand_emits_stresses(cls: str) -> None:
    comps = expand_shorthand("shell_resultant", ndm=3, ndf=6)
    assert comps == _SHELL_LAYOUTS[(cls, "stress")]
    rec = _gauss(comps, cls)
    assert mpco_ops_args([rec]) == ("mpco", "run.mpco", "-E", "stresses")
    assert emit_mpco_tcl([rec]) == "recorder mpco run.mpco -E stresses"
    assert emit_mpco_python([rec]) == (
        "ops.recorder('mpco', 'run.mpco', '-E', 'stresses')")


def test_mpco_line_station_bending_moment_stays_section_force() -> None:
    """The topology split: a beam ``bending_moment`` is ``section.force``."""
    rec = ResolvedRecorderRecord(
        category="line_stations", name="b",
        components=("axial_force", "bending_moment_y", "bending_moment_z"),
        dt=None, n_steps=None, element_ids=np.array([1]),
    )
    assert collect_mpco_tokens([rec]) == ([], ["section.force"])


def test_mpco_gauss_derived_scalar_records_raw_tensor() -> None:
    assert collect_mpco_tokens([_gauss(("von_mises_stress",))]) == (
        [], ["stresses"])


@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_mpco_unmapped_gauss_component_raises(emit) -> None:
    rec = _gauss(("stress_xx", "unrouted_quantity_xx"), name="bad")
    with pytest.raises(ValueError, match=r"'bad'.*'unrouted_quantity_xx'"):
        emit([rec])


# =====================================================================
# Gauss material state, plastic strain and the Truss axial force
# =====================================================================
#
# Oracle: the MPCO material-state reader recovers a canonical's parent
# (``parent_token_for_canonical``) and walks the groups
# ``mpco_gauss_group_aliases`` lists for the parent's keyword; derived
# plastic scalars are computed from the ``plastic_strain_*`` tensor.
# Before this, ``equivalent_plastic_strain*`` emitted ``-E strains`` (the
# total-strain bucket: an empty read) and ``damage*`` / ``plastic_strain_*``
# raised. The live read-back is in
# tests/opensees/integration_ladruno/test_results_catalog_hdf5_live.py.

#: The gauss vocabulary, composed as ``recorder._CATEGORY_CANONICALS``.
_GAUSS_VOCABULARY = sorted(set(
    STRESS + STRAIN + PLASTIC_STRAIN + DERIVED_SCALARS + MATERIAL_STATE))

_MATERIAL_STATE_EXPECTED = {
    "damage": "material.damage",
    "damage_tension": "material.damage",
    "damage_compression": "material.damage",
    "equivalent_plastic_strain": "material.equivalentPlasticStrain",
    "equivalent_plastic_strain_tension": "material.equivalentPlasticStrain",
    "equivalent_plastic_strain_compression": "material.equivalentPlasticStrain",
}


def _read_side_groups(canonical: str) -> tuple[str, ...]:
    """Every MPCO group the reader walks for ``canonical``."""
    if canonical in DERIVED_PLASTIC_STRAIN_SCALARS:
        parent: str | None = "plastic_strain"
    else:
        parent = parent_token_for_canonical(canonical)
    keyword = gauss_keyword_for_canonical(parent or canonical)
    assert keyword is not None, canonical
    return mpco_gauss_group_aliases(keyword)


@pytest.mark.parametrize("canonical", sorted(_MATERIAL_STATE_EXPECTED))
@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_mpco_material_state_emits_material_group(emit, canonical: str) -> None:
    token = _MATERIAL_STATE_EXPECTED[canonical]
    assert token in _read_side_groups(canonical)
    out = emit([_gauss((canonical,))])
    if emit is mpco_ops_args:
        assert out == ("mpco", "run.mpco", "-E", token)
    elif emit is emit_mpco_tcl:
        assert out == f"recorder mpco run.mpco -E {token}"
    else:
        assert out == f"ops.recorder('mpco', 'run.mpco', '-E', '{token}')"


@pytest.mark.parametrize(
    "canonical", list(PLASTIC_STRAIN) + list(DERIVED_PLASTIC_STRAIN_SCALARS))
def test_mpco_plastic_strain_emits_material_plastic_strain(canonical: str) -> None:
    """Includes ``equivalent_plastic_strain_current``: derived from the
    tensor, not the material's accumulated PEEQ."""
    assert "material.plasticStrain" in _read_side_groups(canonical)
    assert collect_mpco_tokens([_gauss((canonical,))]) == (
        [], ["material.plasticStrain"])


@pytest.mark.parametrize("canonical", _GAUSS_VOCABULARY)
def test_every_gauss_canonical_emits_a_group_the_reader_walks(
    canonical: str,
) -> None:
    """The card's done-when: no gauss canonical leaves an MPCO emit silently."""
    _node, tokens = collect_mpco_tokens([_gauss((canonical,))])
    assert len(tokens) == 1, (canonical, tokens)
    if tokens[0] in ("stresses", "strains"):
        # Tensor components route by keyword; derived scalars record the
        # tensor they are computed from.
        return
    assert tokens[0] in _read_side_groups(canonical), (canonical, tokens)


@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_mpco_gauss_truss_axial_force_raises(emit) -> None:
    """MPCO's ``axialForce`` bucket has no GAUSS_IDS; the reader refuses it."""
    rec = _gauss(("axial_force",), cls="Truss", name="bar")
    with pytest.raises(ValueError, match=r"'bar'.*'axial_force'.*GAUSS_IDS"):
        emit([rec])


# =====================================================================
# Line stations: section forces record, section deformations refuse (#2025)
# =====================================================================
#
# Oracle: the MPCO reader finds a line-stations canonical under the
# bucket ``canonical_to_line_station_token`` names, and decodes a bucket
# only when every META component name is in ``MPCO_SECTION_NAME_TO_CODE``
# (the section-force names). A live fork run (PR #2025 body) wrote
# ``section.deformation`` with META ``eps,kappaZ,kappaY,theta,gammaY,
# gammaZ``: the read of ``curvature_z`` raised on ``eps``. Without that
# bucket, every deformation read came back ``(1, 0)``. Before #2025 the
# emit wrote no token for ``axial_strain`` / ``torsional_strain`` /
# ``curvature_*`` and sent ``shear_strain_*`` to ``section.force``
# (prefix ``shear``): an empty read, no warning.

#: MPCO's META names for a section.deformation bucket (live, fork 79e062367).
_MPCO_SECTION_DEFORMATION_NAMES = frozenset(
    {"eps", "kappaZ", "kappaY", "theta", "gammaY", "gammaZ"})


def _line(components, name="beam") -> ResolvedRecorderRecord:
    return ResolvedRecorderRecord(
        category="line_stations", name=name, components=tuple(components),
        dt=None, n_steps=None, element_ids=np.array([1]),
    )


def test_line_station_vocabularies_are_not_vacuous() -> None:
    assert len(LINE_DIAGRAMS) == 6
    assert set(LINE_STATION_DEFORMATIONS) == {
        "axial_strain", "torsional_strain", "curvature_y", "curvature_z",
        "shear_strain_y", "shear_strain_z"}


@pytest.mark.xfail(
    strict=True,
    reason="#2053 taught the MPCO reader the section.deformation names; "
           "#2056 switches the emit to record section.deformation and "
           "retires this tripwire",
)
def test_mpco_reader_still_cannot_decode_section_deformation() -> None:
    """Tripwire: when the reader learns these names, emit section.deformation."""
    assert _MPCO_SECTION_DEFORMATION_NAMES.isdisjoint(MPCO_SECTION_NAME_TO_CODE)


@pytest.mark.parametrize("canonical", LINE_DIAGRAMS)
@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_mpco_line_station_force_emits_section_force(emit, canonical: str) -> None:
    assert canonical_to_line_station_token(canonical) == (
        "section.force", "section_force")
    out = emit([_line((canonical,))])
    if emit is mpco_ops_args:
        assert out == ("mpco", "run.mpco", "-E", "section.force")
    elif emit is emit_mpco_tcl:
        assert out == "recorder mpco run.mpco -E section.force"
    else:
        assert out == "ops.recorder('mpco', 'run.mpco', '-E', 'section.force')"


@pytest.mark.parametrize("canonical", LINE_STATION_DEFORMATIONS)
@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_mpco_line_station_deformation_raises(emit, canonical: str) -> None:
    # The reader routes it to section.deformation, a bucket it cannot decode.
    assert canonical_to_line_station_token(canonical) == (
        "section.deformation", "section_deformation")
    with pytest.raises(
        ValueError,
        match=rf"'kappa_beam'.*line_stations.*'{canonical}'.*section\.deformation",
    ):
        emit([_line((canonical,), name="kappa_beam")])


@pytest.mark.parametrize("emit", [mpco_ops_args, emit_mpco_tcl, emit_mpco_python])
def test_mpco_line_station_mixed_record_raises_not_partial(emit) -> None:
    """A force + deformation record refuses whole, never emits section.force alone."""
    rec = _line(LINE_DIAGRAMS + LINE_STATION_DEFORMATIONS, name="mixed")
    with pytest.raises(ValueError, match=r"'mixed'.*'axial_strain'"):
        emit([rec])


def test_mpco_line_station_unrouted_component_raises() -> None:
    assert canonical_to_line_station_token("stress_xx") is None
    with pytest.raises(ValueError, match=r"'beam'.*'stress_xx'.*no line-stations"):
        collect_mpco_tokens([_line(("axial_force", "stress_xx"))])

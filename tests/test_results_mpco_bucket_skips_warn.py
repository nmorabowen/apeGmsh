"""Every bucket the MPCO reader skips is named, or classified as expected (#2047).

``_mpco.py`` used to wrap six per-bucket decodes in ``except ValueError:
continue``: a refused bucket (malformed META, mixed layer counts, several
sub-GPs per layer, a ``NUM_COLUMNS`` mismatch) vanished from the listing
and from the read without a word. Each case below builds one synthetic
bucket whose decoder raises ``ValueError`` at one of those sites and
asserts the read names it, by bucket key and reason, in one warning:

========================================  ===============================
site                                      warning
========================================  ===============================
``_gauss_available_components``           :class:`MPCOBucketSkippedWarning`
``_read_gauss_material_state``            :class:`MPCOBucketSkippedWarning`
``_springs_available_components``         :class:`MPCOBucketSkippedWarning`
``read_springs``                          :class:`MPCOBucketSkippedWarning`
``_layers_available_components``          :class:`LayerBucketUnreadWarning`
``read_layers``                           :class:`LayerBucketUnreadWarning`
========================================  ===============================

The layer level keeps the class #2091 introduced, so one warning per read
names both the buckets discovery leaves out and the ones the read refuses.
An expected skip (a spring index the bucket lacks, a component absent from
the bucket's META) is decided by an explicit predicate and stays silent.
The #2093 items (mixed layer counts, sub-GPs, the warning's wording for
non-shell classes, one warning per listing) are pinned here as well.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from apeGmsh.opensees._response_catalog import (
    ELE_TAG_ASDShellQ4,
    ELE_TAG_FourNodeTetrahedron,
    IntRule,
    is_fiber_catalogued,
    is_layer_catalogued,
)
from apeGmsh.results import Results
from apeGmsh.results.readers import _mpco_layer_io as _mlayer
from apeGmsh.results.readers._mpco import MPCOBucketSkippedWarning
from apeGmsh.results.readers._mpco_layer_io import LayerBucketUnreadWarning
from apeGmsh.results.readers._protocol import ResultLevel
from tests.conftest import _stub_model_h5_path
from tests.test_results_mpco_layer_mock import (
    _add_layer_bucket,
    _add_layered_section,
    _add_shell_connectivity,
    _create_skeleton as _layer_skeleton,
)
from tests.test_results_mpco_material_mock import (
    _add_material_state_bucket,
    _add_tet_connectivity,
    _create_skeleton as _material_skeleton,
)
from tests.test_results_mpco_spring_mock import _make_spring_mpco

_TET_KEY = (
    f"{ELE_TAG_FourNodeTetrahedron}-FourNodeTetrahedron[{IntRule.Tet_GL_1}:0:0]"
)
_SHELL_KEY = f"{ELE_TAG_ASDShellQ4}-ASDShellQ4[{IntRule.Quad_GL_2}:0:0]"


def _open(path: Path) -> Results:
    return Results.from_mpco(str(path), model_h5=_stub_model_h5_path())


def _only(record: "list[warnings.WarningMessage]", category: type) -> list[str]:
    return [str(w.message) for w in record if issubclass(w.category, category)]


# =====================================================================
# Gauss material state: _gauss_available_components, _read_gauss_material_state
# =====================================================================

def _damage_file(tmp_path: Path, *, num_columns: int) -> Path:
    """One FourNodeTetrahedron ``d+,d-`` bucket; ``num_columns`` 2 is
    well formed, any other width is refused by the resolver."""
    path = tmp_path / "damage.mpco"
    f, stage_name = _material_skeleton(
        path, node_ids=np.arange(1, 5, dtype=np.int64),
        coords=np.zeros((4, 3)), n_steps=1, dt=0.1,
    )
    stage = f[stage_name]
    eids = np.array([10, 11], dtype=np.int64)
    _add_tet_connectivity(stage, element_ids=eids)
    _add_material_state_bucket(
        stage, element_ids=eids, n_components=2,
        component_symbols=("d+", "d-"),
        flat_data=np.ones((1, 2, 2)), token="material.damage",
    )
    bucket = stage[f"RESULTS/ON_ELEMENTS/material.damage/{_TET_KEY}"]
    bucket.attrs["NUM_COLUMNS"] = np.array([num_columns], dtype=np.int32)
    f.close()
    return path


class TestGaussMaterialState:
    def test_listing_names_a_refused_bucket(self, tmp_path: Path) -> None:
        r = _open(_damage_file(tmp_path, num_columns=5))
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                comps = r._reader.available_components(
                    r._reader.stages()[0].id, ResultLevel.GAUSS,
                )
        finally:
            r._reader.close()
        assert "damage_tension" not in comps
        (msg,) = _only(rec, MPCOBucketSkippedWarning)
        assert f"material.damage/{_TET_KEY}" in msg
        assert "NUM_COLUMNS=5" in msg

    def test_read_names_a_refused_bucket(self, tmp_path: Path) -> None:
        r = _open(_damage_file(tmp_path, num_columns=5))
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.gauss.get(component="damage_tension")
        finally:
            r._reader.close()
        assert slab.values.shape[1] == 0
        (msg,) = _only(rec, MPCOBucketSkippedWarning)
        assert "'damage_tension'" in msg
        assert f"material.damage/{_TET_KEY}" in msg
        assert "NUM_COLUMNS=5" in msg

    def test_well_formed_bucket_reads_in_silence(self, tmp_path: Path) -> None:
        r = _open(_damage_file(tmp_path, num_columns=2))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MPCOBucketSkippedWarning)
                assert "damage_tension" in r._reader.available_components(
                    r._reader.stages()[0].id, ResultLevel.GAUSS,
                )
                slab = r.elements.gauss.get(component="damage_tension")
        finally:
            r._reader.close()
        assert slab.values.shape == (1, 2)


# =====================================================================
# Springs: _springs_available_components, read_springs
# =====================================================================

def _spring_file(tmp_path: Path, *, num_columns: int) -> Path:
    """A three-spring ZeroLength bucket; ``num_columns`` 3 is well formed."""
    path = _make_spring_mpco(
        tmp_path / "springs.mpco", element_ids=[1, 2], n_springs=3,
        force_values=np.arange(12, dtype=np.float64).reshape(2, 2, 3),
    )
    with h5py.File(path, "r+") as f:
        f["MODEL_STAGE[1]/RESULTS/ON_ELEMENTS/basicForce/19-ZeroLength[1:0:0]"
          ].attrs["NUM_COLUMNS"] = num_columns
    return path


class TestSprings:
    def test_listing_names_a_refused_bucket(self, tmp_path: Path) -> None:
        r = _open(_spring_file(tmp_path, num_columns=7))
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                comps = r._reader.available_components(
                    r._reader.stages()[0].id, ResultLevel.SPRINGS,
                )
        finally:
            r._reader.close()
        assert comps == []
        (msg,) = _only(rec, MPCOBucketSkippedWarning)
        assert "basicForce/19-ZeroLength[1:0:0]" in msg
        assert "NUM_COLUMNS=7" in msg

    def test_read_names_a_refused_bucket(self, tmp_path: Path) -> None:
        r = _open(_spring_file(tmp_path, num_columns=7))
        try:
            with pytest.warns(MPCOBucketSkippedWarning) as rec:
                slab = r.elements.springs.get(component="spring_force_1")
        finally:
            r._reader.close()
        assert slab.values.shape[1] == 0
        (msg,) = _only(rec, MPCOBucketSkippedWarning)
        assert "'spring_force_1'" in msg
        assert "basicForce/19-ZeroLength[1:0:0]" in msg

    def test_absent_spring_index_is_an_expected_skip(
        self, tmp_path: Path,
    ) -> None:
        """``spring_force_3`` on a three-spring bucket: the decoder's own
        predicate (``spring_idx >= n_springs``) skips it, in silence."""
        r = _open(_spring_file(tmp_path, num_columns=3))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MPCOBucketSkippedWarning)
                slab = r.elements.springs.get(component="spring_force_3")
                good = r.elements.springs.get(component="spring_force_1")
        finally:
            r._reader.close()
        assert slab.values.shape[1] == 0
        np.testing.assert_array_equal(good.values, [[1.0, 4.0], [7.0, 10.0]])


# =====================================================================
# Layers: _layers_available_components, read_layers (+ #2093 items)
# =====================================================================

def _layer_file(
    tmp_path: Path, *,
    layer_counts: tuple[int, int] = (2, 2),
    sub_gps: int = 1,
    extra_unread: "tuple[str, ...]" = (),
) -> Path:
    """Two ASDShellQ4 elements (20, 21) in one bucket.

    ``layer_counts`` gives each element's layered section; unequal counts
    are the mixed bucket. ``sub_gps`` > 1 writes the columns of several
    sub-GPs per layer. ``extra_unread`` adds a bucket of a class neither
    level reads, under each named token group.
    """
    path = tmp_path / "layers.mpco"
    f, stage_name = _layer_skeleton(
        path, node_ids=np.arange(1, 9, dtype=np.int64),
        coords=np.zeros((8, 3)), n_steps=1, dt=1.0,
    )
    stage = f[stage_name]
    eids = np.array([20, 21], dtype=np.int64)
    _add_shell_connectivity(
        stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
        surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
        element_ids=eids,
    )
    for tag, (eid, n_lyr) in enumerate(zip(eids, layer_counts), start=1):
        _add_layered_section(
            stage, section_tag=tag,
            section_class="LayeredShellFiberSection",
            element_sgp_pairs=[(int(eid), g) for g in range(4)],
            layer_thickness=np.full(n_lyr, 0.01),
            layer_material=np.ones(n_lyr, dtype=np.int64),
        )
    n_layers = layer_counts[0]
    width = 4 * n_layers * sub_gps
    _add_layer_bucket(
        stage, class_tag=ELE_TAG_ASDShellQ4, class_name="ASDShellQ4",
        surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
        element_ids=eids, n_sgp=4, n_layers=n_layers,
        flat_data=np.ones((1, 2, width)),
    )
    stage[f"RESULTS/ON_ELEMENTS/material.fiber.stress/{_SHELL_KEY}"].attrs[
        "NUM_COLUMNS"] = np.array([width], dtype=np.int32)
    for token in extra_unread:
        _add_layer_bucket(
            stage, class_tag=999, class_name="WeirdShell",
            surface_int_rule=IntRule.Quad_GL_2, custom_rule_idx=0,
            element_ids=np.array([30]), n_sgp=4, n_layers=1,
            flat_data=np.zeros((1, 1, 4)), token=token,
        )
    f.close()
    return path


def _layer_listing(path: Path) -> "tuple[list[str], list[str]]":
    r = _open(path)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            comps = r._reader.available_components(
                r._reader.stages()[0].id, ResultLevel.LAYERS,
            )
    finally:
        r._reader.close()
    return comps, _only(rec, LayerBucketUnreadWarning)


def _layer_read(path: Path, **kw) -> "tuple[object, list[str]]":
    r = _open(path)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            slab = r._reader.read_layers(
                r._reader.stages()[0].id, "fiber_stress", **kw,
            )
    finally:
        r._reader.close()
    return slab, _only(rec, LayerBucketUnreadWarning)


class TestLayers:
    def test_mixed_layer_counts_listing_warns(self, tmp_path: Path) -> None:
        comps, msgs = _layer_listing(_layer_file(tmp_path, layer_counts=(2, 3)))
        assert comps == []
        (msg,) = msgs
        assert f"material.fiber.stress/{_SHELL_KEY}" in msg
        assert "different layer counts [2, 3]" in msg

    def test_mixed_layer_counts_read_warns(self, tmp_path: Path) -> None:
        slab, msgs = _layer_read(_layer_file(tmp_path, layer_counts=(2, 3)))
        assert slab.values.shape[1] == 0
        (msg,) = msgs
        assert f"material.fiber.stress/{_SHELL_KEY}" in msg
        assert "different layer counts [2, 3]" in msg

    def test_several_sub_gps_per_layer_warn(self, tmp_path: Path) -> None:
        """The ``NUM_COLUMNS`` check: the listing applies it as the read
        does, so neither lists nor reads the bucket in silence."""
        path = _layer_file(tmp_path, sub_gps=2)
        comps, listing = _layer_listing(path)
        slab, reading = _layer_read(path)
        assert comps == []
        assert slab.values.shape[1] == 0
        for msgs in (listing, reading):
            (msg,) = msgs
            assert _SHELL_KEY in msg
            assert "NUM_COLUMNS=16" in msg

    def test_read_names_discovery_and_read_skips_in_one_warning(
        self, tmp_path: Path,
    ) -> None:
        path = _layer_file(
            tmp_path, layer_counts=(2, 3),
            extra_unread=("material.fiber.stress",),
        )
        _, msgs = _layer_read(path)
        (msg,) = msgs
        assert msg.startswith("2 MPCO layer bucket(s) left unread")
        assert "999-WeirdShell" in msg
        assert "different layer counts" in msg

    def test_listing_warns_once_across_both_tokens(
        self, tmp_path: Path,
    ) -> None:
        """#2093 item 3: the listing walks fiber_stress and fiber_strain;
        an unread bucket under each token used to warn once per token."""
        path = _layer_file(
            tmp_path,
            extra_unread=("material.fiber.stress", "material.fiber.strain"),
        )
        comps, msgs = _layer_listing(path)
        assert comps == ["fiber_stress"]
        (msg,) = msgs
        assert "material.fiber.stress/999-WeirdShell" in msg
        assert "material.fiber.strain/999-WeirdShell" in msg

    def test_layer_index_beyond_the_bucket_is_named(
        self, tmp_path: Path,
    ) -> None:
        slab, msgs = _layer_read(
            _layer_file(tmp_path), layer_indices=np.array([5]),
        )
        assert slab.values.shape[1] == 0
        (msg,) = msgs
        assert "layer_indices [5] out of range [0, 2)" in msg

    def test_well_formed_bucket_reads_in_silence(self, tmp_path: Path) -> None:
        path = _layer_file(tmp_path)
        comps, listing = _layer_listing(path)
        slab, reading = _layer_read(path)
        assert comps == ["fiber_stress"]
        assert slab.values.shape == (1, 16)
        assert listing == reading == []


# =====================================================================
# #2093 item 3: the unread-bucket reason fits non-shell classes
# =====================================================================

def _classify(key: str) -> "str | None":
    reason, _, _ = _mlayer._classify_layer_bucket(key, "fiber_stress")
    return reason


class TestUnreadReasonWording:
    def test_zero_length_section_is_named_by_both_catalogs(self) -> None:
        assert not is_layer_catalogued("ZeroLengthSection", "fiber_stress")
        assert not is_fiber_catalogued("ZeroLengthSection", "fiber_stress")
        reason = _classify("165-ZeroLengthSection[1:0:0]")
        assert reason == (
            "class 'ZeroLengthSection' has no layered-shell entry "
            "(LAYER_CATALOG) and no beam fiber-section entry (FIBER_CATALOG)"
        )

    def test_beam_on_a_standard_rule_is_named_as_a_beam(self) -> None:
        assert is_fiber_catalogued("ForceBeamColumn3d", "fiber_stress")
        reason = _classify("74-ForceBeamColumn3d[2:0:0]")
        assert reason is not None
        assert "'ForceBeamColumn3d' is a beam fiber-section class" in reason
        assert "layered-shell" not in reason
        assert "this bucket has rule 2" in reason

    def test_beam_on_a_custom_rule_belongs_to_the_fiber_level(self) -> None:
        assert _classify("74-ForceBeamColumn3d[1000:0:0]") is None

"""Viewer scene id-space regression — paired opens (ADR 0043 slice 1.3).

The reader fix (2d159362) attaches the fem_eid↔ops-tag translator for
ANY paired model, so unfiltered element/gauss reads return
``element_index`` in **fem_eid** space. The viewer's scene join table
(``element_id_to_cell``) is built from whatever :class:`ViewerData` the
resolver picks — pre-fix, for a paired ``.mpco``/``.ladruno`` open with
no ``fem=``, that was the reader's **ops-tag-keyed** embedded snapshot:
every id of a translated read then missed (or mis-hit) the scene table,
silently blanking or mis-colouring contours, GP markers and thresholds.

This file pins the invariant with the same synthetic offset pairing as
``tests/test_results_tag_offset_uncomposed.py``: the id universe of an
UNFILTERED gauss read must be a subset of the scene's element-id
universe, resolved exactly the way
:meth:`ResultsViewer._build_viewer_data` does
(``resolve_orientation_source`` → ``ViewerData.from_h5``, else
``ViewerData.from_fem``). No Qt / VTK involved — ``element_id_to_cell``
keys are exactly the ViewerData element-group ids
(``fem_scene.py`` builds the table from them), so the assertion runs
headless.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from apeGmsh.opensees._response_catalog import IntRule, flatten, lookup
from apeGmsh.results import Results
from apeGmsh.viewers.data import ViewerData
from apeGmsh.viewers.data._h5_probe import resolve_orientation_source

# Reuse the offset-pairing builders from the reader regression suite —
# same synthetic MPCO bucket, same offset fem_eids (501/502 vs ops 1/2).
from tests.test_results_mpco_element_mock import (
    _add_stress_bucket,
    _create_mpco_skeleton,
)
from tests.test_results_tag_offset_uncomposed import (
    _build_offset_uncomposed_model_h5,
)


def _write_offset_mpco(tmp_path: Path, ops_tags: list[int]) -> Path:
    """A gauss stress bucket keyed by ops tags (what the recorder writes)."""
    layout = lookup("FourNodeTetrahedron", IntRule.Tet_GL_1, "stress")
    mpco = tmp_path / "offset_scene.mpco"
    f, stage_name = _create_mpco_skeleton(
        mpco,
        node_ids=np.array([1, 2, 3, 4, 5], dtype=np.int64),
        node_coords=np.eye(5, 3),
        n_steps=2,
        dt=0.5,
    )
    try:
        T, E = 2, len(ops_tags)
        per_comp = {
            name: (
                np.arange(E, dtype=np.float64).reshape(1, E, 1) * 100.0
                + np.arange(T, dtype=np.float64).reshape(T, 1, 1) * 10.0
                + float(k)
            )
            for k, name in enumerate(layout.component_layout)
        }
        flat = flatten(per_comp, layout)
        _add_stress_bucket(
            f[stage_name],
            bracket_key=f"{layout.class_tag}-FourNodeTetrahedron[300:0:0]",
            class_tag=layout.class_tag,
            int_rule=IntRule.Tet_GL_1,
            element_ids=np.array(ops_tags, dtype=np.int64),
            flat_data=flat,
            n_gauss_points=1,
        )
    finally:
        f.close()
    return mpco


def _scene_element_ids(view: ViewerData) -> set[int]:
    """The element-id universe the scene joins against —
    ``fem_scene.build_fem_scene`` builds ``element_id_to_cell`` from
    exactly these group ids."""
    ids: set[int] = set()
    for group in view.elements:
        ids.update(int(e) for e in group.ids)
    return ids


def _offset_case(tmp_path):
    """(mpco, model_h5) with offset fem_eids and a verified precondition."""
    model_h5, ops_to_fem = _build_offset_uncomposed_model_h5(tmp_path)
    assert all(tag != fem for tag, fem in ops_to_fem.items()), (
        "fixture precondition: ops tags must differ from offset fem_eids"
    )
    mpco = _write_offset_mpco(tmp_path, sorted(ops_to_fem))
    return mpco, model_h5


def test_paired_open_resolves_scene_source_to_model_h5(tmp_path):
    """Paired open → the scene source is the fem_eid-keyed model.h5,
    not the reader's ops-tag-keyed embedded snapshot."""
    mpco, model_h5 = _offset_case(tmp_path)
    with Results.from_mpco(mpco, model_h5=model_h5) as r:
        assert resolve_orientation_source(r) == Path(model_h5)


def test_stub_paired_open_keeps_the_embedded_snapshot_scene():
    """An UNRELATED stub ``model_h5=`` (the sanctioned test-suite
    pattern) attaches a pairing that never fires: reads stay in the
    file's own ops-tag space, so the probe must NOT switch the scene to
    the stub's mesh — the embedded snapshot stays the consistent
    choice. Guards the adversarial finding against the attachment-based
    gate: with ``_tag_map is not None`` as the gate, the viewer
    rendered the stub's 1-element mesh against the real file's ids.
    """
    import pytest

    fixture = Path("tests/fixtures/results/elasticFrame.mpco")
    if not fixture.exists():
        pytest.skip(f"Missing fixture: {fixture}")
    from tests.conftest import _stub_model_h5_path

    with Results.from_mpco(fixture, model_h5=_stub_model_h5_path()) as r:
        assert resolve_orientation_source(r) is None, (
            "a stub pairing that cannot cover the file's element ids "
            "must not promote the stub model.h5 to scene source"
        )
        # And the fallback scene really does cover the untranslated
        # reads — the pre-existing, self-consistent pairing.
        view = ViewerData.from_fem(r.fem)
        scene_ids = _scene_element_ids(view)
        slab = r.stage(r.stages[0].id).elements.line_stations.get(
            component="axial_force",
        )
        read_ids = set(int(e) for e in slab.element_index)
        assert read_ids and read_ids <= scene_ids


def test_paired_open_scene_ids_cover_unfiltered_gauss_read(tmp_path):
    """Paired open, offset fem_eids, no ``fem=`` — the viewer-resolved
    scene must speak the same id space the translated reads return.

    Pre-fix (``resolve_orientation_source`` probing only ``_path``),
    the scene came from the ops-tag-keyed embedded snapshot while the
    unfiltered read returned fem_eids: the subset assertion fails and
    every cell silently drops out of the contour join. Deliberately no
    assertion on ``source`` itself here — however the resolver decides,
    the id spaces must agree.
    """
    mpco, model_h5 = _offset_case(tmp_path)
    with Results.from_mpco(mpco, model_h5=model_h5) as r:
        # Mirror ResultsViewer._build_viewer_data exactly.
        source = resolve_orientation_source(r)
        view = (
            ViewerData.from_h5(str(source))
            if source is not None
            else ViewerData.from_fem(r.fem)
        )
        scene_ids = _scene_element_ids(view)

        slab = r.stage(r.stages[0].id).elements.gauss.get(
            component="stress_xx",
        )
        read_ids = set(int(e) for e in slab.element_index)
        assert read_ids, "unfiltered gauss read returned no elements"
        assert read_ids <= scene_ids, (
            f"unfiltered gauss read returned element ids {sorted(read_ids)}"
            f" that the viewer scene (ids {sorted(scene_ids)}) cannot "
            "join — contours/markers/thresholds would silently blank or "
            "mis-colour"
        )
